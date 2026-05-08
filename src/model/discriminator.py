import torch
import torch.nn as nn


# =========================
# PATCH DISCRIMINATOR
#
# Phán đoán từng patch 70×70 thay vì toàn ảnh
# → ổn định hơn standard GAN
# → phù hợp SR vì tập trung vào texture cục bộ
#
# Input:  [B, 3, 512, 512]
# Output: [B, 1, 30, 30]  — mỗi cell = 1 patch prediction
# =========================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            # Layer 1: không norm (input layer)
            block(in_channels,          base_channels,     stride=2, norm=False),  # 512→256
            block(base_channels,        base_channels * 2, stride=2),              # 256→128
            block(base_channels * 2,    base_channels * 4, stride=2),              # 128→64
            block(base_channels * 4,    base_channels * 8, stride=1),              # 64→63
            # Output: 1 channel prediction map
            nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1),               # 63→62
        )

        # Khởi tạo weights chuẩn cho GAN
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)