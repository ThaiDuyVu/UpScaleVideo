import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision.models import vgg16

from src.data.dataset import VideoDataset
from src.model.generator import Generator


# =========================
# CUDA OPTIMIZATION
# =========================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# =========================
# DEVICE
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("=" * 60)
print("🔥 Using device:", device)

if torch.cuda.is_available():
    print("🔥 GPU:", torch.cuda.get_device_name(0))
    print(f"🔥 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("=" * 60)


# =========================
# DATASET
# =========================
dataset = VideoDataset("data/lr_frames", "data/frames")

print(f"✅ Dataset size: {len(dataset)}")

train_size = int(0.9 * len(dataset))
val_size   = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])


# =========================
# HIGH PERFORMANCE DATALOADER
# Optimized for V100
# =========================
train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_set,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)


# =========================
# DEBUG DATA
# =========================
lr_sample, hr_sample = next(iter(train_loader))

print("=" * 60)
print(f"[DEBUG] LR range: {lr_sample.min().item():.4f} -> {lr_sample.max().item():.4f}")
print(f"[DEBUG] HR range: {hr_sample.min().item():.4f} -> {hr_sample.max().item():.4f}")
print(f"[DEBUG] LR shape: {lr_sample.shape}")
print(f"[DEBUG] HR shape: {hr_sample.shape}")
print("=" * 60)

DATA_MAX = lr_sample.max().item()

if DATA_MAX > 1.5:
    print("[WARNING] Auto normalize /255 enabled")
    NORMALIZE = True
else:
    print("[OK] Dataset already normalized")
    NORMALIZE = False


# =========================
# PERCEPTUAL LOSS
# =========================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        try:
            vgg = vgg16(weights="DEFAULT").features[:16].eval()
            print("✅ VGG16 pretrained loaded")
        except Exception as e:
            print(f"[WARNING] Cannot load pretrained VGG ({e})")
            vgg = vgg16(weights=None).features[:16].eval()

        for p in vgg.parameters():
            p.requires_grad = False

        self.vgg = vgg

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


perceptual_loss_fn = PerceptualLoss().to(device)


# =========================
# MODEL
# =========================
model = Generator().to(device)

# FIX: AdamW + weight_decay thay Adam thuần
# AdamW tách weight decay ra khỏi gradient update → ổn định hơn
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# FIX: ReduceLROnPlateau thay StepLR
# Giảm LR khi val_loss không cải thiện sau patience=2 epoch
# thay vì giảm cứng mỗi 10 epoch bất kể model học tốt hay không
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=True
)

criterion = nn.L1Loss()

# AMP scaler — chỉ active trên CUDA, tắt trên MPS/CPU
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())


# =========================
# CHECKPOINT
# =========================
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")


# =========================
# METRICS
# =========================
def psnr(pred, target):
    pred   = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


# =========================
# EDGE LOSS
# =========================
def gradient_loss(pred, target):
    pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


# =========================
# KL LOSS WITH FREE BITS
# Không phạt nếu KL < free_bits → tránh posterior collapse
# =========================
def kl_loss_free_bits(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()


# =========================
# TRAIN LOOP
# =========================
epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss    = 0
    valid_batches = 0  # FIX: đếm batch hợp lệ, tránh NaN ảnh hưởng avg loss

    # KL warm-up: 5 epoch đầu tắt để model học reconstruction trước
    # Sau đó tăng dần từ 1e-7, tối đa 1e-5
    if epoch < 5:
        kl_weight = 0.0
    else:
        kl_weight = min(1e-7 * (epoch - 4), 1e-5)

    loop = tqdm(train_loader)

    for lr_img, hr_img in loop:

        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        if NORMALIZE:
            lr_img = lr_img / 255.0
            hr_img = hr_img / 255.0

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

            pred, mu, logvar = model(lr_img)

            # FIX: clamp logvar/mu trước khi tính loss
            # logvar > 88 → exp() → inf trên float32 → NaN
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            mu     = torch.clamp(mu,     min=-10.0, max=10.0)

            pred = pred.clamp(0, 1)

            loss_l1   = criterion(pred, hr_img)
            loss_edge = gradient_loss(pred, hr_img)
            loss_perc = perceptual_loss_fn(pred, hr_img)
            kl_loss   = kl_loss_free_bits(mu, logvar)

            loss = (
                1.0  * loss_l1
                + 0.1  * loss_edge
                + 0.05 * loss_perc
                + kl_weight * kl_loss
            )

        # FIX: skip batch NaN thay vì crash cả training
        if not torch.isfinite(loss):
            print(f"\n[WARNING] NaN/Inf loss — skipping batch | "
                  f"l1={loss_l1.item():.4f} edge={loss_edge.item():.4f} "
                  f"perc={loss_perc.item():.4f} kl={kl_loss.item():.4f}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # FIX: clip 0.5 thay 1.0 — kiểm soát gradient explosion tốt hơn
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        valid_batches += 1

        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(
            total=f"{loss.item():.4f}",
            l1=f"{loss_l1.item():.4f}",
            edge=f"{loss_edge.item():.4f}",
            perc=f"{loss_perc.item():.4f}",
            kl=f"{kl_loss.item():.2f}",
            kl_w=f"{kl_weight:.2e}"
        )

    # FIX: chia valid_batches thực tế thay len(train_loader)
    avg_train_loss = total_loss / max(valid_batches, 1)

    print("\n" + "=" * 60)
    print(f"🔥 Train Loss:    {avg_train_loss:.6f}")
    print(f"🔥 Valid batches: {valid_batches}/{len(train_loader)}")
    print("=" * 60)


    # =========================
    # VALIDATION
    # =========================
    model.eval()

    val_loss    = 0
    psnr_list   = []
    ssim_list   = []
    num_batches = 0

    MAX_VAL_BATCHES = 20

    with torch.no_grad():

        for i, (lr_img, hr_img) in enumerate(val_loader):

            if i >= MAX_VAL_BATCHES:
                break

            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            if NORMALIZE:
                lr_img = lr_img / 255.0
                hr_img = hr_img / 255.0

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred, _, _ = model(lr_img)
                pred = pred.clamp(0, 1)
                val_loss += criterion(pred, hr_img).item()

            psnr_list.append(psnr(pred, hr_img).item())

            # FIX: tính SSIM toàn bộ batch, không chỉ i==0
            for b in range(pred.shape[0]):
                pred_np = pred[b].permute(1, 2, 0).cpu().numpy()
                hr_np   = hr_img[b].permute(1, 2, 0).cpu().numpy()

                s = ssim(
                    hr_np, pred_np,
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_list.append(s)

            num_batches += 1

    # FIX: chia đúng num_batches thực tế
    val_loss /= num_batches

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"🔥 Val Loss:      {val_loss:.6f}")
    print(f"🔥 PSNR:          {avg_psnr:.2f} dB")
    print(f"🔥 SSIM:          {avg_ssim:.4f}")
    print(f"🔥 KL weight:     {kl_weight:.2e}")


    # =========================
    # LR SCHEDULER
    # FIX: ReduceLROnPlateau cần val_loss để quyết định giảm LR
    # =========================
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"🔥 Learning rate: {current_lr:.2e}")


    # =========================
    # SAVE CHECKPOINT
    # =========================
    torch.save(
        model.state_dict(),
        f"checkpoints/epoch_{epoch+1}.pth"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"✅ BEST MODEL SAVED (val_loss={best_val_loss:.6f})")