import torch
from src.model.wavelet import DWT, IWT

x = torch.rand(1, 3, 128, 128)

dwt = DWT()
iwt = IWT()

y = dwt(x)
x_recon = iwt(y)

print("Input:", x.shape)
print("After DWT:", y.shape)
print("Reconstructed:", x_recon.shape)

# Check error
error = torch.mean((x - x_recon)**2)
print("Reconstruction error:", error.item())