import torch
import torch.nn as nn
import torch.nn.functional as F


class FastKANConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, grid_size=16):
        super(FastKANConv2d, self).__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.grid_size    = grid_size

        # Base conv (linear part)
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # Grid parameters (learn nonlinear mapping)
        self.grid = nn.Parameter(torch.linspace(-1, 1, grid_size), requires_grad=False)

        # Coefficients for interpolation
        self.coeff = nn.Parameter(torch.randn(out_channels, in_channels, grid_size) * 0.01)
        #                                                                              ^^^^
        # FIX: khởi tạo nhỏ → nonlinear term gần 0 lúc đầu
        # → model học từ base_conv trước, sau đó dần học nonlinear

        # FIX: scale của gaussian — dùng learnable thay vì cứng = grid_size
        # grid_size=16 quá lớn → exp(-16*dist) collapse về 0 với dist > 0.5
        self.log_scale = nn.Parameter(torch.tensor(2.0))  # scale = exp(2) ≈ 7.4

    def forward(self, x):
        B, C, H, W = x.shape

        # Base linear output
        base = self.base_conv(x)

        # Normalize input to [-1,1]
        x_norm = torch.tanh(x)

        # Expand for grid matching
        x_exp = x_norm.unsqueeze(-1)            # [B, C, H, W, 1]
        grid  = self.grid.view(1, 1, 1, 1, -1)  # [1, 1, 1, 1, G]

        # Distance to grid points
        dist = torch.abs(x_exp - grid)           # [B, C, H, W, G] — luôn >= 0

        # FIX: dùng learnable scale thay vì cứng grid_size
        # clamp log_scale tránh scale quá lớn gây vanishing
        scale   = torch.exp(torch.clamp(self.log_scale, min=0.5, max=3.0))
        weights = torch.exp(-dist * scale)       # [B, C, H, W, G]

        # Reshape: [B, C, H, W, G] → [B, C, G, H, W]
        weights = weights.permute(0, 1, 4, 2, 3)

        # Nonlinear output: [B, out_channels, H, W]
        nonlinear = torch.einsum("b c g h w, o c g -> b o h w", weights, self.coeff)

        return base + nonlinear


# ===== KAN BLOCK =====
class KANBlock(nn.Module):
    def __init__(self, channels):
        super(KANBlock, self).__init__()

        self.kan1 = FastKANConv2d(channels, channels)
        self.norm1 = nn.GroupNorm(num_groups=4, num_channels=channels)  # FIX: thêm norm

        self.kan2 = FastKANConv2d(channels, channels)
        self.norm2 = nn.GroupNorm(num_groups=4, num_channels=channels)  # FIX: thêm norm

    def forward(self, x):
        residual = x

        x = self.kan1(x)
        x = self.norm1(x)          # FIX: normalize trước activation
        x = F.leaky_relu(x, 0.2)  # FIX: LeakyReLU thay ReLU

        x = self.kan2(x)
        x = self.norm2(x)

        return x + residual


# ===== STACKED KAN =====
class KAN(nn.Module):
    def __init__(self, channels=12, num_blocks=4):
        super(KAN, self).__init__()

        self.blocks = nn.Sequential(
            *[KANBlock(channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)