import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletEncoder(nn.Module):
    """
    Deterministic encoder thay VAE stochastic.
    - Không reparameterize → không noise → SR output sắc nét hơn
    - Không KL loss → không conflict với reconstruction loss
    - Giữ skip connection để gradient flow tốt
    """
    def __init__(self, in_channels=12, latent_dim=1024):
        super(WaveletEncoder, self).__init__()

        self.in_channels = in_channels
        self.latent_dim  = latent_dim

        # ===== ENCODER =====
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),   # 64 → 32
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),            # 32 → 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),           # 16 → 8
            nn.LeakyReLU(0.2),
            nn.Flatten()                                           # → 128*8*8 = 8192
        )

        self.encoder_norm = nn.LayerNorm(128 * 8 * 8)
        self.fc_encode    = nn.Linear(128 * 8 * 8, latent_dim)

        # ===== DECODER =====
        self.fc_decode = nn.Linear(latent_dim, 128 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),          # 8 → 16
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),           # 16 → 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),  # 32 → 64
        )

        # Skip connection: giữ lại wavelet gốc, encoder chỉ học phần chênh lệch
        self.skip_proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B = x.size(0)

        enc   = self.encoder(x)
        enc   = self.encoder_norm(enc)
        z     = self.fc_encode(enc)          # deterministic, không sample

        dec   = self.fc_decode(z)
        dec   = dec.view(B, 128, 8, 8)
        recon = self.decoder(dec)

        # Skip connection
        recon = recon + self.skip_proj(x)

        return recon                          # chỉ trả recon, không mu/logvar