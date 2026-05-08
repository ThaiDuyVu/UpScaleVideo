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
        #  → DWT      → [B,12,64,64]   wavelet domain
        #  → Encoder  → [B,12,64,64]   gated residual (bypass-safe)
        #  → KAN      → [B,12,64,64]   multi-scale feature refine
        #  → IWT      → [B,3,128,128]  pixel domain
        #  → refine   → [B,3,128,128]  residual refine
        #  → up1 (2x) → [B,3,256,256]
        #  → up2 (2x) → [B,3,512,512]
        #  → final    → [B,3,512,512]  output [0,1]
        # ===================================================

        # Refine sau IWT — học residual so với LR gốc
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
            nn.Conv2d(32, 12, 1),      # 12 = 3 * 2^2
            nn.PixelShuffle(2),        # → [B, 3, 256, 256]
        )

        # Upscale 256 → 512 (2x)
        self.up2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 12, 1),      # 12 = 3 * 2^2
            nn.PixelShuffle(2),        # → [B, 3, 512, 512]
        )

        # Refine cuối — KHÔNG dùng Sigmoid để tránh bị kéo về 0.5
        # Thay bằng clamp [0,1] trong forward
        self.final_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        # Init upscale conv tránh checkerboard
        self._init_subpixel(self.up1)
        self._init_subpixel(self.up2)

        # Init refine: weight nhỏ, bias=0 → output ≈ identity lúc đầu
        # KHÔNG zero weight (sẽ gây output hằng số)
        nn.init.xavier_uniform_(self.refine[-1].weight, gain=0.1)
        nn.init.zeros_(self.refine[-1].bias)

        nn.init.xavier_uniform_(self.final_refine[-1].weight, gain=0.1)
        nn.init.zeros_(self.final_refine[-1].bias)

    def _init_subpixel(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, 128, 128]

        wave     = self.dwt(x)           # [B, 12, 64, 64]
        recon    = self.encoder(wave)    # [B, 12, 64, 64] — gated bypass-safe
        features = self.kan(recon)       # [B, 12, 64, 64]
        pix      = self.iwt(features)    # [B, 3, 128, 128]

        # Residual: shortcut từ LR gốc + phần refine từ IWT output
        # x đi thẳng → model chỉ học "phần dư" cần thêm
        pix = x + self.refine(pix)       # [B, 3, 128, 128]

        pix = self.up1(pix)              # [B, 3, 256, 256]
        pix = self.up2(pix)              # [B, 3, 512, 512]

        # Refine cuối + clamp thay Sigmoid
        # Sigmoid gây output ≈ 0.5 khi weight random → PSNR 4dB
        out = self.final_refine(pix)     # [B, 3, 512, 512]
        out = out.clamp(0, 1)            # range [0,1]

        return out