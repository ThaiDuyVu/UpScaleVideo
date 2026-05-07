import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()

    def forward(self, x):
        # Pixel unshuffle (downscale)
        x = F.pixel_unshuffle(x, 2)  # [B, 4C, H/2, W/2]

        # Haar transform (linear combination)
        B, C, H, W = x.shape
        x = x.view(B, -1, 4, H, W)

        LL = (x[:, :, 0] + x[:, :, 1] + x[:, :, 2] + x[:, :, 3]) / 2
        LH = (-x[:, :, 0] - x[:, :, 1] + x[:, :, 2] + x[:, :, 3]) / 2
        HL = (-x[:, :, 0] + x[:, :, 1] - x[:, :, 2] + x[:, :, 3]) / 2
        HH = (x[:, :, 0] - x[:, :, 1] - x[:, :, 2] + x[:, :, 3]) / 2

        out = torch.cat([LL, LH, HL, HH], dim=1)
        return out


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        x = x.view(B, 4, C//4, H, W)

        LL = x[:,0]
        LH = x[:,1]
        HL = x[:,2]
        HH = x[:,3]

        x0 = (LL - LH - HL + HH) / 2
        x1 = (LL - LH + HL - HH) / 2
        x2 = (LL + LH - HL - HH) / 2
        x3 = (LL + LH + HL + HH) / 2

        out = torch.stack([x0, x1, x2, x3], dim=2)
        out = out.view(B, -1, H, W)

        # Pixel shuffle (upscale)
        out = F.pixel_shuffle(out, 2)

        return out