import torch
from src.model.kan import KAN

x = torch.randn(1, 12, 64, 64)

model = KAN()

out = model(x)

print("Input:", x.shape)
print("Output:", out.shape)