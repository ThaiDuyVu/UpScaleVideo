import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.wavelet import DWT, IWT
from src.model.vae import WaveletEncoder   # đổi tên import
from src.model.kan import KAN


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dwt     = DWT()
        self.encoder = WaveletEncoder()    # thay self.vae
        self.kan     = KAN()
        self.iwt     = IWT()

        # Refine sau IWT
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
        )

        # Upscale 128 → 256 (2x) — thêm 2 conv trước PixelShuffle
        self.up1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 12, 1),           # 12 = 3 * 2^2
            nn.PixelShuffle(2),
        )

        # Upscale 256 → 512 (2x) — thêm 2 conv trước PixelShuffle
        self.up2 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 12, 1),
            nn.PixelShuffle(2),
        )

        # Refine cuối
        self.final_refine = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

        # Init sub-pixel conv tránh checkerboard
        self._init_subpixel(self.up1)
        self._init_subpixel(self.up2)

    def _init_subpixel(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 3, 128, 128]

        wave     = self.dwt(x)              # [B, 12, 64, 64]
        recon    = self.encoder(wave)       # [B, 12, 64, 64] — không có mu/logvar
        features = self.kan(recon)          # [B, 12, 64, 64]
        pix      = self.iwt(features)       # [B, 3, 128, 128]

        pix = pix + self.refine(pix)        # residual refine

        pix = self.up1(pix)                 # [B, 3, 256, 256]
        pix = self.up2(pix)                 # [B, 3, 512, 512]

        out = self.final_refine(pix)        # [B, 3, 512, 512]

        return out                          # chỉ trả out, không mu/logvar