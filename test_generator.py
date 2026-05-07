import torch
from src.model.generator import Generator

x = torch.randn(1, 3, 128, 128)

model = Generator()

out, mu, logvar = model(x)

print("Input:", x.shape)
print("Output:", out.shape)