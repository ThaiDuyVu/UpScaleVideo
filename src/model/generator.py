import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.wavelet import DWT, IWT
from src.model.vae import WaveletEncoder
from src.model.kan import KAN


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dwt     = DWT()
        self.encoder = WaveletEncoder()
        self.kan     = KAN()
        self.iwt     = IWT()

        # ===================================================
        # PIPELINE:
        #
        #  LR [B,3,128,128]
        #  → DWT     → [B,12,64,64]   wavelet domain
        #  → Encoder → [B,12,64,64]   gated residual (bypass-safe)
        #  → KAN     → [B,12,64,64]   multi-scale feature refine
        #  → IWT     → [B,3,128,128]  pixel domain
        #  → refine  → [B,3,128,128]  residual refine (+ LR shortcut)
        #  → up1     → [B,3,256,256]  upsample 2x
        #  → up2     → [B,3,512,512]  upsample 2x
        #  → final   → [B,3,512,512]  output [0,1]
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
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 12, 1),
            nn.PixelShuffle(2),             # → [B, 3, 256, 256]
        )

        # Upscale 256 → 512 (2x)
        self.up2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 12, 1),
            nn.PixelShuffle(2),             # → [B, 3, 512, 512]
        )

        # Refine cuối
        self.final_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        # Init upscale conv — tránh checkerboard artifact
        self._init_subpixel(self.up1)
        self._init_subpixel(self.up2)

        # Init refine cuối: bias → 0.5 để Sigmoid output ≈ 0.5 lúc đầu
        # Tránh output bị kéo về 0 hoặc 1 hoàn toàn
        nn.init.zeros_(self.final_refine[-2].weight)
        nn.init.constant_(self.final_refine[-2].bias, 0.0)

    def _init_subpixel(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, 128, 128]

        wave     = self.dwt(x)           # [B, 12, 64, 64]
        recon    = self.encoder(wave)    # [B, 12, 64, 64] — gated, bypass-safe
        features = self.kan(recon)       # [B, 12, 64, 64]
        pix      = self.iwt(features)    # [B, 3, 128, 128]

        # Residual với LR input gốc — bảo toàn signal, model chỉ học phần dư
        pix = x + self.refine(pix)       # [B, 3, 128, 128]

        pix = self.up1(pix)              # [B, 3, 256, 256]
        pix = self.up2(pix)              # [B, 3, 512, 512]

        out = self.final_refine(pix)     # [B, 3, 512, 512], range [0,1]

        return out