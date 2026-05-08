import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, in_channels=12, latent_dim=1024):
        # FIX: latent_dim 256 → 1024
        # 8192 → 256 là nén 32:1, quá mạnh cho SR task
        # 8192 → 1024 là nén 8:1, giữ được nhiều tần số cao hơn
        super(VAE, self).__init__()

        self.in_channels = in_channels
        self.latent_dim  = latent_dim

        # ===== ENCODER =====
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 64 → 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 32 → 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # 16 → 8
            nn.LeakyReLU(0.2),
            nn.Flatten()                                          # → 128*8*8 = 8192
        )

        self.encoder_norm = nn.LayerNorm(128 * 8 * 8)

        self.fc_mu     = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # ===== DECODER =====
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # 8 → 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),           # 16 → 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 32 → 64
        )

        # FIX: skip connection — VAE chỉ học "phần chênh lệch"
        # gradient flow tốt hơn, học nhanh hơn
        self.skip_proj = nn.Conv2d(in_channels, in_channels, 1)  # 1x1 conv

        # Init fc_logvar nhỏ → logvar ≈ 0 lúc đầu → std ≈ 1
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.xavier_uniform_(self.fc_logvar.weight, gain=0.01)

    def reparameterize(self, mu, logvar):
        logvar_clamped = torch.clamp(logvar, min=-4.0, max=4.0)
        std = torch.exp(0.5 * logvar_clamped)
        eps = torch.randn_like(std)
        return mu + eps * std, logvar_clamped

    def forward(self, x):
        B = x.size(0)

        enc = self.encoder(x)
        enc = self.encoder_norm(enc)

        mu     = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        mu     = torch.clamp(mu, min=-10.0, max=10.0)

        z, logvar = self.reparameterize(mu, logvar)

        dec = self.fc_decode(z)
        dec = dec.view(B, 128, 8, 8)
        recon = self.decoder(dec)

        # Skip connection: giữ lại thông tin wavelet gốc
        recon = recon + self.skip_proj(x)

        return recon, mu, logvar