import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.wavelet import DWT, IWT
from src.model.vae import VAE
from src.model.kan import KAN


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dwt = DWT()
        self.vae = VAE()
        self.kan = KAN()
        self.iwt = IWT()

        # ===================================================
        # FLOW MỚI (nhất quán wavelet domain):
        #
        #  LR [B,3,128,128]
        #  → DWT  → [B,12,64,64]   wavelet coefficients
        #  → VAE  → [B,12,64,64]   vẫn wavelet domain
        #  → KAN  → [B,12,64,64]   vẫn wavelet domain
        #  → IWT  → [B,3,128,128]  pixel domain (reconstruct)
        #  → up1  → [B,3,256,256]  upsample 2x
        #  → up2  → [B,3,512,512]  upsample 2x
        #
        # FIX: IWT phải nhận wavelet data → đặt TRƯỚC các upscale block
        # ===================================================

        # Refine sau IWT trong pixel domain
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        # Upscale 128 → 256 (2x)
        self.up1 = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),             # 12ch → 3ch, H×2, W×2
        )

        # Upscale 256 → 512 (2x)
        self.up2 = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(2),             # 12ch → 3ch, H×2, W×2
        )

        # Refine cuối
        self.final_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),                   # output [0,1]
        )

    def forward(self, x):
        # x: [B, 3, 128, 128]

        # ===== WAVELET DOMAIN =====
        wave = self.dwt(x)              # [B, 12, 64, 64]

        recon, mu, logvar = self.vae(wave)   # [B, 12, 64, 64]

        features = self.kan(recon)      # [B, 12, 64, 64]

        # ===== BACK TO PIXEL DOMAIN =====
        # IWT nhận đúng wavelet coefficients → pixel domain
        pix = self.iwt(features)        # [B, 3, 128, 128]

        # Refine + residual connection với LR input
        pix = pix + self.refine(pix)    # [B, 3, 128, 128]

        # ===== UPSCALE =====
        pix = self.up1(pix)             # [B, 3, 256, 256]
        pix = self.up2(pix)             # [B, 3, 512, 512]

        out = self.final_refine(pix)    # [B, 3, 512, 512], range [0,1]

        return out, mu, logvar