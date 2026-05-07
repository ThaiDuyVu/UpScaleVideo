import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== FAST KAN CONV =====
class FastKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, grid_size=16):
        super(FastKANConv2d, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.grid_size    = grid_size

        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.grid      = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)
        self.coeff     = nn.Parameter(torch.randn(out_channels, in_channels, grid_size) * 0.01)
        self.log_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, x):
        base   = self.base_conv(x)
        x_norm = torch.tanh(x)
        x_exp  = x_norm.unsqueeze(-1)
        grid   = self.grid.view(1, 1, 1, 1, -1)
        dist   = torch.abs(x_exp - grid)
        scale  = torch.exp(torch.clamp(self.log_scale, min=0.5, max=3.0))
        weights = torch.exp(-dist * scale)
        weights = weights.permute(0, 1, 4, 2, 3)
        nonlinear = torch.einsum("b c g h w, o c g -> b o h w", weights, self.coeff)
        return base + nonlinear


# ===== CBAM CHANNEL ATTENTION =====
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(B, C))
        mx  = self.fc(self.max_pool(x).view(B, C))
        scale = self.sigmoid(avg + mx).view(B, C, 1, 1)
        return x * scale


# ===== CBAM SPATIAL ATTENTION =====
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


# ===== KAN BLOCK (multi-scale + CBAM) =====
class KANBlock(nn.Module):
    def __init__(self, channels):
        super(KANBlock, self).__init__()

        # Multi-scale: song song 3x3 và 5x5
        self.kan1_3x3 = FastKANConv2d(channels, channels, kernel_size=3, padding=1)
        self.kan1_5x5 = FastKANConv2d(channels, channels, kernel_size=5, padding=2)
        self.merge    = nn.Conv2d(channels * 2, channels, kernel_size=1)  # fuse 2 nhánh
        self.norm1    = nn.GroupNorm(num_groups=4, num_channels=channels)

        self.kan2  = FastKANConv2d(channels, channels)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)

        # CBAM sau kan2
        self.cbam_c = ChannelAttention(channels)
        self.cbam_s = SpatialAttention()

    def forward(self, x):
        residual = x

        # Multi-scale branch
        x3 = self.kan1_3x3(x)
        x5 = self.kan1_5x5(x)
        x  = self.merge(torch.cat([x3, x5], dim=1))
        x  = self.norm1(x)
        x  = F.leaky_relu(x, 0.2)

        # Second KAN
        x = self.kan2(x)
        x = self.norm2(x)

        # CBAM attention
        x = self.cbam_c(x)
        x = self.cbam_s(x)

        return x + residual


# ===== STACKED KAN (tăng từ 4 → 6 blocks) =====
class KAN(nn.Module):
    def __init__(self, channels=12, num_blocks=6):
        super(KAN, self).__init__()

        self.blocks = nn.Sequential(
            *[KANBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)