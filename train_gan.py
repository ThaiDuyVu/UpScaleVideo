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
from src.model.discriminator import PatchDiscriminator


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
NORMALIZE = DATA_MAX > 1.5
print("[WARNING] Auto normalize enabled" if NORMALIZE else "[OK] Dataset already normalized")


# =========================
# PERCEPTUAL LOSS (VGG)
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
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std
        return F.l1_loss(self.vgg(pred_n), self.vgg(target_n))

perceptual_loss_fn = PerceptualLoss().to(device)


# =========================
# MODELS
# =========================
generator     = Generator().to(device)
discriminator = PatchDiscriminator().to(device)


# =========================
# LOAD PRETRAINED GENERATOR
# Tiếp tục từ best_model.pth của bước 1
# Không train lại từ đầu — tiết kiệm thời gian
# =========================
PRETRAINED_PATH = "checkpoints/best_model.pth"

if os.path.exists(PRETRAINED_PATH):
    generator.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
    print(f"✅ Loaded pretrained generator: {PRETRAINED_PATH}")
else:
    print(f"[WARNING] No pretrained found at {PRETRAINED_PATH} — training from scratch")


# =========================
# OPTIMIZERS
# Generator: lr thấp hơn vì đã pretrained
# Discriminator: lr cao hơn một chút để bắt kịp generator
# =========================
opt_G = torch.optim.AdamW(
    generator.parameters(),
    lr=5e-5,           # thấp hơn train_warmup (1e-4) vì đã pretrained
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

opt_D = torch.optim.AdamW(
    discriminator.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# Scheduler cho cả 2
scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_G, mode='min', factor=0.5, patience=3, min_lr=1e-6
)
scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_D, mode='min', factor=0.5, patience=3, min_lr=1e-6
)

criterion_pixel = nn.L1Loss()
criterion_gan   = nn.BCEWithLogitsLoss()  # PatchGAN dùng BCEWithLogits

# AMP scalers — 1 cho G, 1 cho D
scaler_G = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
scaler_D = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())


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

def gradient_loss(pred, target):
    pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

def kl_loss_free_bits(mu, logvar, free_bits=0.5):
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_per_dim, min=free_bits).mean()


# =========================
# CHECKPOINT + RESUME
# =========================
os.makedirs("checkpoints", exist_ok=True)
best_val_loss = float("inf")
start_epoch   = 0

# Resume từ epoch 3 (epoch cuối chạy tốt trước khi crash)
# Thay số này nếu muốn resume từ epoch khác
RESUME_PATH = "checkpoints/gan_epoch_3.pth"

if os.path.exists(RESUME_PATH):
    ckpt = torch.load(RESUME_PATH, map_location=device)
    generator.load_state_dict(ckpt["generator"])
    discriminator.load_state_dict(ckpt["discriminator"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    start_epoch   = ckpt["epoch"] + 1
    best_val_loss = ckpt["val_loss"]
    print(f"✅ Resumed from {RESUME_PATH} (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f})")
else:
    print(f"[INFO] No resume checkpoint found, starting from epoch 1")


# =========================
# GAN TRAINING LOOP
# =========================
epochs = 30

for epoch in range(start_epoch, epochs):

    generator.train()
    discriminator.train()

    # KL weight (tiếp tục từ epoch 16 của bước 1)
    # Giữ kl_weight nhỏ — VAE đã học tốt rồi
    kl_weight = 1e-5

    # GAN weight warm-up (conservative):
    # adv loss ~8-9 rất lớn → cần weight rất nhỏ để không át L1
    # Công thức: effective_adv = adv * gan_weight
    # Muốn effective_adv < 10% của L1 (~0.02) → gan_weight < 0.002
    #
    # Epoch 0-4:  0.001  (ổn định)
    # Epoch 5-9:  0.0015 (tăng rất chậm)
    # Epoch 10-14: 0.002 (plateau)
    # Epoch 15+:  0.003  (max)
    if epoch < 5:
        gan_weight = 0.001
    elif epoch < 10:
        gan_weight = 0.0015
    elif epoch < 15:
        gan_weight = 0.002
    else:
        gan_weight = 0.003

    total_loss_G  = 0
    total_loss_D  = 0
    valid_batches = 0

    loop = tqdm(train_loader)

    for lr_img, hr_img in loop:

        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)

        if NORMALIZE:
            lr_img = lr_img / 255.0
            hr_img = hr_img / 255.0

        B = lr_img.size(0)

        # Labels cho PatchGAN
        # Dùng label smoothing: real=0.9 thay 1.0 → ổn định hơn
        real_label = torch.ones (B, 1, 62, 62, device=device) * 0.9
        fake_label = torch.zeros(B, 1, 62, 62, device=device)

        # =========================================
        # BƯỚC 1: TRAIN DISCRIMINATOR
        # D cần phân biệt real HR vs fake SR
        # =========================================
        opt_D.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):

            # Generate fake SR (không cần gradient cho G ở bước này)
            with torch.no_grad():
                fake_sr, _, _ = generator(lr_img)
                fake_sr = fake_sr.clamp(0, 1)

            # D nhận real HR → predict real
            pred_real = discriminator(hr_img)
            loss_D_real = criterion_gan(pred_real, real_label)

            # D nhận fake SR → predict fake
            pred_fake = discriminator(fake_sr.detach())  # detach: không update G
            loss_D_fake = criterion_gan(pred_fake, fake_label)

            # Tổng D loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5

        scaler_D.scale(loss_D).backward()
        scaler_D.unscale_(opt_D)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler_D.step(opt_D)
        scaler_D.update()


        # =========================================
        # BƯỚC 2: TRAIN GENERATOR
        # G cần: (1) đánh lừa D + (2) pixel accurate
        # =========================================
        opt_G.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):

            fake_sr, mu, logvar = generator(lr_img)

            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            mu     = torch.clamp(mu,     min=-10.0, max=10.0)
            fake_sr = fake_sr.clamp(0, 1)

            # Adversarial loss: G muốn D nghĩ fake là real
            pred_fake_for_G = discriminator(fake_sr)
            loss_G_adv = criterion_gan(pred_fake_for_G, real_label)

            # Reconstruction losses
            loss_l1   = criterion_pixel(fake_sr, hr_img)
            loss_edge = gradient_loss(fake_sr, hr_img)
            loss_perc = perceptual_loss_fn(fake_sr, hr_img)
            loss_kl   = kl_loss_free_bits(mu, logvar)

            # Tổng G loss
            # L1 vẫn dominant để giữ pixel accuracy
            # GAN thêm texture sắc nét
            loss_G = (
                1.0        * loss_l1
                + 0.1      * loss_edge
                + 0.05     * loss_perc
                + gan_weight * loss_G_adv
                + kl_weight * loss_kl
            )

        if not torch.isfinite(loss_G):
            print(f"\n[WARNING] NaN G loss — skipping | "
                  f"l1={loss_l1.item():.4f} adv={loss_G_adv.item():.4f}")
            opt_G.zero_grad(set_to_none=True)
            continue

        scaler_G.scale(loss_G).backward()
        scaler_G.unscale_(opt_G)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=0.5)
        scaler_G.step(opt_G)
        scaler_G.update()

        total_loss_G  += loss_G.item()
        total_loss_D  += loss_D.item()
        valid_batches += 1

        loop.set_description(f"GAN Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(
            G=f"{loss_G.item():.4f}",
            D=f"{loss_D.item():.4f}",
            l1=f"{loss_l1.item():.4f}",
            adv=f"{loss_G_adv.item():.4f}",
            g_w=f"{gan_weight:.3f}",
        )

    avg_G = total_loss_G / max(valid_batches, 1)
    avg_D = total_loss_D / max(valid_batches, 1)

    print("\n" + "=" * 60)
    print(f"🔥 Generator Loss:     {avg_G:.6f}")
    print(f"🔥 Discriminator Loss: {avg_D:.6f}")
    print(f"🔥 GAN weight:         {gan_weight:.4f}")
    print(f"🔥 Valid batches:      {valid_batches}/{len(train_loader)}")
    print("=" * 60)


    # =========================
    # VALIDATION
    # =========================
    generator.eval()

    val_loss  = 0
    psnr_list = []
    ssim_list = []
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

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred, _, _ = generator(lr_img)
                pred = pred.clamp(0, 1)
                val_loss += criterion_pixel(pred, hr_img).item()

            psnr_list.append(psnr(pred, hr_img).item())

            for b in range(pred.shape[0]):
                pred_np = pred[b].permute(1, 2, 0).cpu().numpy()
                hr_np   = hr_img[b].permute(1, 2, 0).cpu().numpy()
                s = ssim(hr_np, pred_np, channel_axis=2, data_range=1.0)
                ssim_list.append(s)

            num_batches += 1

    val_loss /= num_batches
    avg_psnr  = sum(psnr_list) / len(psnr_list)
    avg_ssim  = sum(ssim_list) / len(ssim_list)

    print(f"🔥 Val Loss:  {val_loss:.6f}")
    print(f"🔥 PSNR:      {avg_psnr:.2f} dB")
    print(f"🔥 SSIM:      {avg_ssim:.4f}")

    # =========================
    # SCHEDULER
    # =========================
    scheduler_G.step(val_loss)
    scheduler_D.step(val_loss)
    print(f"🔥 LR G: {opt_G.param_groups[0]['lr']:.2e}  |  LR D: {opt_D.param_groups[0]['lr']:.2e}")

    # =========================
    # SAVE CHECKPOINT
    # =========================
    torch.save({
        "generator":     generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "opt_G":         opt_G.state_dict(),
        "opt_D":         opt_D.state_dict(),
        "epoch":         epoch,
        "val_loss":      val_loss,
    }, f"checkpoints/gan_epoch_{epoch+1}.pth")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(generator.state_dict(), "checkpoints/best_gan_model.pth")
        print(f"✅ BEST GAN MODEL SAVED (val_loss={best_val_loss:.6f})")