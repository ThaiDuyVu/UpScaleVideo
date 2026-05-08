import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# RESIDUAL BLOCK
# Dùng trong encoder/decoder để tăng capacity
# mà không làm mất gradient
# =========================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================
# VAE - FULLY CONVOLUTIONAL
#
# THAY ĐỔI SO VỚI CŨ:
# ❌ Cũ: encoder flatten → FC(8192→256) → FC(256→8192) → decoder
#         Phá vỡ spatial structure, mất tần số cao
#
# ✅ Mới: encoder Conv stride=2 → latent Conv 1x1 → decoder ConvTranspose
#         Giữ nguyên spatial map [B,C,H,W] xuyên suốt
#         Không có FC layer nào → không mất spatial information
#
# Spatial flow:
#   Input:   [B, 12, 64, 64]
#   Enc1:    [B, 32, 32, 32]  (stride=2)
#   Enc2:    [B, 64, 16, 16]  (stride=2)
#   Latent:  [B, 128, 16, 16] (Conv 1x1, NO spatial compression)
#   mu/lv:   [B, 64, 16, 16]  (Conv 1x1)
#   z:       [B, 64, 16, 16]
#   Dec1:    [B, 64, 32, 32]  (ConvTranspose stride=2)
#   Dec2:    [B, 32, 64, 64]  (ConvTranspose stride=2)
#   Output:  [B, 12, 64, 64]  (Conv 1x1)
# =========================
class VAE(nn.Module):
    def __init__(self, in_channels=12, latent_channels=64):
        super().__init__()

        self.in_channels     = in_channels
        self.latent_channels = latent_channels

        # ===== ENCODER =====
        # Stride=2 conv giữ spatial structure (không flatten)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64→32
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(32),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 32→16
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )

        # Latent projection — Conv 1x1, không phá spatial
        self.latent_proj = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # mu và logvar giữ nguyên spatial map [B, latent_channels, H, W]
        self.conv_mu     = nn.Conv2d(128, latent_channels, 1)
        self.conv_logvar = nn.Conv2d(128, latent_channels, 1)

        # Init logvar → 0 lúc đầu → std ≈ 1 → training ổn định
        nn.init.zeros_(self.conv_logvar.bias)
        nn.init.xavier_uniform_(self.conv_logvar.weight, gain=0.01)

        # ===== DECODER =====
        self.dec_proj = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 1),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16→32
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(64),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32→64
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(32),
        )

        self.dec_out = nn.Conv2d(32, in_channels, 1)

        # Skip connection: VAE học residual, không học toàn bộ
        # → gradient flow tốt hơn, converge nhanh hơn
        self.skip_proj = nn.Conv2d(in_channels, in_channels, 1)

    # =========================
    # REPARAMETERIZATION
    # Spatial-aware: z có shape [B, latent_channels, H, W]
    # thay vì [B, latent_dim] như cũ
    # =========================
    def reparameterize(self, mu, logvar):
        # Clamp logvar: [-4, 4] → std trong [0.135, 7.39]
        logvar = torch.clamp(logvar, min=-4.0, max=4.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar

    def forward(self, x):
        # x: [B, 12, 64, 64] — wavelet coefficients

        # ===== ENCODE =====
        e1 = self.enc1(x)           # [B, 32, 32, 32]
        e2 = self.enc2(e1)          # [B, 64, 16, 16]
        h  = self.latent_proj(e2)   # [B, 128, 16, 16]

        mu     = self.conv_mu(h)                              # [B, 64, 16, 16]
        logvar = self.conv_logvar(h)                          # [B, 64, 16, 16]
        mu     = torch.clamp(mu, min=-10.0, max=10.0)

        z, logvar = self.reparameterize(mu, logvar)           # [B, 64, 16, 16]

        # ===== DECODE =====
        d  = self.dec_proj(z)       # [B, 128, 16, 16]
        d  = self.dec1(d)           # [B, 64, 32, 32]
        d  = self.dec2(d)           # [B, 32, 64, 64]
        recon = self.dec_out(d)     # [B, 12, 64, 64]

        # Skip connection: output = reconstruction + input gốc
        recon = recon + self.skip_proj(x)

        return recon, mu, logvar