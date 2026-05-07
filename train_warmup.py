import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from torchvision.models import vgg16

from src.data.dataset import VideoDataset
from src.model.generator import Generator


# =========================
# DEVICE SETUP (M1 SAFE)
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =========================
# DATASET
# =========================
dataset = VideoDataset("data/lr_frames", "data/frames")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_set,
    batch_size=2,
    num_workers=0,
    pin_memory=False
)


# =========================
# DEBUG: KIỂM TRA DATA RANGE
# Nếu max > 1.0 → cần normalize trong dataset.py
# =========================
lr_sample, hr_sample = next(iter(train_loader))
print("=" * 50)
print(f"[DEBUG] LR range: min={lr_sample.min().item():.4f}, max={lr_sample.max().item():.4f}")
print(f"[DEBUG] HR range: min={hr_sample.min().item():.4f}, max={hr_sample.max().item():.4f}")
print(f"[DEBUG] LR shape: {lr_sample.shape}")
print(f"[DEBUG] HR shape: {hr_sample.shape}")
print("=" * 50)

# Nếu data không nằm trong [0,1], normalize tại đây
# (Tốt hơn nên fix trong dataset.py, nhưng đây là safety net)
DATA_MAX = lr_sample.max().item()
if DATA_MAX > 1.5:
    print("[WARNING] Data range > 1.0 → sẽ normalize /255 trong training loop")
    NORMALIZE = True
else:
    print("[OK] Data đã ở range [0,1]")
    NORMALIZE = False


# =========================
# PERCEPTUAL LOSS (VGG)
# Fix SSL certificate error trên macOS Python 3.12
# =========================
import ssl
import urllib.request

# Bypass SSL verification khi download VGG weights (chỉ cần 1 lần)
_original_https_open = urllib.request.HTTPSHandler
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))
urllib.request.install_opener(opener)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            vgg = vgg16(weights="DEFAULT").features[:16].eval()
            print("[OK] VGG16 pretrained weights loaded")
        except Exception as e:
            print(f"[WARNING] Không tải được VGG weights ({e})")
            print("[WARNING] Dùng VGG16 không có pretrained — perceptual loss kém hơn nhưng vẫn chạy được")
            vgg = vgg16(weights=None).features[:16].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg

        # VGG normalization (ImageNet mean/std)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))


perceptual_loss_fn = PerceptualLoss().to(device)


# =========================
# MODEL
# =========================
model = Generator().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

criterion = nn.L1Loss()


# =========================
# CHECKPOINT SETUP
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
# EDGE / GRADIENT LOSS
# =========================
def gradient_loss(pred, target):
    pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

    loss_x = F.l1_loss(pred_dx, target_dx)
    loss_y = F.l1_loss(pred_dy, target_dy)

    return loss_x + loss_y


# =========================
# KL LOSS VỚI FREE BITS
# Tránh KL bùng nổ — không phạt nếu KL < free_bits
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
    total_loss   = 0
    valid_batches = 0  # đếm batch hợp lệ (không NaN)

    # =========================
    # KL WARM-UP (cải tiến)
    # - 5 epoch đầu: tắt hoàn toàn KL để model học reconstruction trước
    # - Sau đó tăng dần rất chậm, giới hạn tối đa ở 1e-5
    # =========================
    if epoch < 5:
        kl_weight = 0.0
    else:
        kl_weight = min(1e-7 * (epoch - 4), 1e-5)

    loop = tqdm(train_loader)

    for lr_img, hr_img in loop:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        # Safety normalize nếu data không ở [0,1]
        if NORMALIZE:
            lr_img = lr_img / 255.0
            hr_img = hr_img / 255.0

        pred, mu, logvar = model(lr_img)

        # ===== NaN GUARD #1: clamp logvar tránh exp() overflow =====
        # logvar.exp() với logvar > 88 → inf trên float32
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        mu     = torch.clamp(mu,     min=-10.0, max=10.0)

        # Clamp output về [0,1] để đảm bảo metric đúng
        pred = pred.clamp(0, 1)

        # ===== LOSS =====
        loss_l1   = criterion(pred, hr_img)
        loss_edge = gradient_loss(pred, hr_img)
        loss_perc = perceptual_loss_fn(pred, hr_img)
        kl_loss   = kl_loss_free_bits(mu, logvar)

        # ===== FINAL LOSS =====
        loss = (
            1.0  * loss_l1
            + 0.1  * loss_edge
            + 0.05 * loss_perc
            + kl_weight * kl_loss
        )

        # ===== NaN GUARD #2: skip batch nếu loss NaN/Inf =====
        if not torch.isfinite(loss):
            print(f"\n[WARNING] NaN/Inf loss — skipping batch | "
                  f"l1={loss_l1.item():.4f} edge={loss_edge.item():.4f} "
                  f"perc={loss_perc.item():.4f} kl={kl_loss.item():.4f}")
            optimizer.zero_grad()
            continue

        # ===== BACKPROP =====
        optimizer.zero_grad()
        loss.backward()

        # Clip chặt hơn để tránh gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

        optimizer.step()

        total_loss   += loss.item()
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

    avg_train_loss = total_loss / max(valid_batches, 1)
    print(f"\nTrain Loss: {avg_train_loss:.6f}  (valid batches: {valid_batches}/{len(train_loader)})")


    # =========================
    # VALIDATION
    # =========================
    model.eval()
    val_loss   = 0
    psnr_list  = []
    ssim_list  = []
    num_batches = 0

    MAX_VAL_BATCHES = 20  # giới hạn để nhanh hơn

    with torch.no_grad():
        for i, (lr_img, hr_img) in enumerate(val_loader):
            if i >= MAX_VAL_BATCHES:
                break

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            if NORMALIZE:
                lr_img = lr_img / 255.0
                hr_img = hr_img / 255.0

            pred, _, _ = model(lr_img)
            pred = pred.clamp(0, 1)

            val_loss += criterion(pred, hr_img).item()
            psnr_list.append(psnr(pred, hr_img).item())

            # FIX: tính SSIM cho TẤT CẢ ảnh trong batch, không chỉ i==0
            for b in range(pred.shape[0]):
                pred_np = pred[b].permute(1, 2, 0).cpu().numpy()
                hr_np   = hr_img[b].permute(1, 2, 0).cpu().numpy()

                s = ssim(
                    hr_np,
                    pred_np,
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_list.append(s)

            num_batches += 1

    val_loss /= num_batches  # FIX: chia đúng số batch thực tế

    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print(f"Val Loss:  {val_loss:.6f}")
    print(f"🔥 PSNR:   {avg_psnr:.2f} dB")
    print(f"🔥 SSIM:   {avg_ssim:.4f}")
    print(f"   KL weight: {kl_weight:.2e}")


    # =========================
    # SCHEDULER
    # =========================
    scheduler.step()


    # =========================
    # SAVE CHECKPOINT
    # =========================
    torch.save(
        model.state_dict(),
        f"checkpoints/epoch_{epoch+1}.pth"
    )


    # =========================
    # BEST MODEL
    # =========================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "checkpoints/best_model.pth")
        print(f"✔ Saved BEST model (val_loss={best_val_loss:.6f})")