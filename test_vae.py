import torch
from src.model.vae import VAE

x = torch.randn(1, 12, 64, 64)

model = VAE()

recon, mu, logvar = model(x)

print("Input:", x.shape)
print("Recon:", recon.shape)
print("Mu:", mu.shape)
print("Logvar:", logvar.shape)