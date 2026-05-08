import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision.models import vgg16
from pytorch_msssim import ssim as ssim_metric

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
# DATALOADER
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
        # FIX: float32 → tránh VGG NaN với AMP float16
        pred_n   = (pred.float()   - self.mean) / self.std
        target_n = (target.float() - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


perceptual_loss_fn = PerceptualLoss().to(device)


# =========================
# MODEL
# =========================
model = Generator().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    # FIX: bỏ verbose=True → dùng get_last_lr() thay thế, tránh FutureWarning
)

criterion = nn.L1Loss()

# FIX: dùng torch.amp.GradScaler thay torch.cuda.amp.GradScaler → tránh FutureWarning
use_amp = torch.cuda.is_available()
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)


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
    mse    = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 10 * torch.log10(1.0 / mse)


# =========================
# LOSS FUNCTIONS
# =========================
def gradient_loss(pred, target):
    """Edge loss."""
    pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def frequency_loss(pred, target):
    """
    FFT Frequency loss.
    FIX: float32 + clamp amplitude → tránh NaN/Inf với AMP float16.
    """
    pred_f     = pred.float().clamp(0, 1)
    target_f   = target.float().clamp(0, 1)
    pred_fft   = torch.fft.rfft2(pred_f,   norm='ortho')
    target_fft = torch.fft.rfft2(target_f, norm='ortho')
    pred_amp   = torch.abs(pred_fft).clamp(max=1e4)
    target_amp = torch.abs(target_fft).clamp(max=1e4)
    return F.l1_loss(pred_amp, target_amp)


def ssim_loss(pred, target):
    """
    SSIM loss.
    FIX: float32 → tránh NaN với AMP float16.
    """
    return 1.0 - ssim_metric(
        pred.float(), target.float(),
        data_range=1.0,
        size_average=True
    )


# =========================
# TRAIN LOOP
# =========================
epochs = 30

for epoch in range(epochs):

    model.train()

    total_loss    = 0
    valid_batches = 0

    loop = tqdm(train_loader)

    for lr_img, hr_img in loop:

        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        if NORMALIZE:
            lr_img = lr_img / 255.0
            hr_img = hr_img / 255.0

        optimizer.zero_grad(set_to_none=True)

        # FIX: dùng torch.amp.autocast thay torch.cuda.amp.autocast → tránh FutureWarning
        with torch.amp.autocast('cuda', enabled=use_amp):

            pred = model(lr_img)

            # Kiểm tra NaN trong output model
            if not torch.isfinite(pred).all():
                print(f"\n[WARNING] NaN/Inf in model output — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            pred = pred.clamp(0, 1)

            loss_l1   = criterion(pred, hr_img)
            loss_edge = gradient_loss(pred, hr_img)
            loss_perc = perceptual_loss_fn(pred, hr_img)
            loss_freq = frequency_loss(pred, hr_img)
            loss_ssim = ssim_loss(pred, hr_img)

            loss = (
                0.7 * loss_l1
                + 0.1 * loss_edge
                + 0.1 * loss_perc
                + 0.1 * loss_freq
                + 0.2 * loss_ssim
            )

        if not torch.isfinite(loss):
            print(f"\n[WARNING] NaN/Inf loss — skipping batch | "
                  f"l1={loss_l1.item():.4f} edge={loss_edge.item():.4f} "
                  f"perc={loss_perc.item():.4f} freq={loss_freq.item():.4f} "
                  f"ssim={loss_ssim.item():.4f}")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

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
            freq=f"{loss_freq.item():.4f}",
            ssim=f"{loss_ssim.item():.4f}",
        )

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

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred     = model(lr_img)
                pred     = pred.clamp(0, 1)
                val_loss += criterion(pred, hr_img).item()

            psnr_list.append(psnr(pred, hr_img).item())

            for b in range(pred.shape[0]):
                # FIX: float() trước numpy → tránh lỗi float16 → numpy
                pred_np = pred[b].permute(1, 2, 0).cpu().float().numpy()
                hr_np   = hr_img[b].permute(1, 2, 0).cpu().float().numpy()

                s = ssim(
                    hr_np, pred_np,
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_list.append(s)

            num_batches += 1

    val_loss /= num_batches
    avg_psnr  = sum(psnr_list) / len(psnr_list)
    avg_ssim  = sum(ssim_list) / len(ssim_list)

    print(f"🔥 Val Loss:      {val_loss:.6f}")
    print(f"🔥 PSNR:          {avg_psnr:.2f} dB")
    print(f"🔥 SSIM:          {avg_ssim:.4f}")

    scheduler.step(val_loss)
    # FIX: dùng get_last_lr() thay verbose=True
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