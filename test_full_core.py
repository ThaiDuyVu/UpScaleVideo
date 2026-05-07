from src.data.dataset import VideoDataset
from torch.utils.data import DataLoader

from src.model.wavelet import DWT
from src.model.vae import VAE
from src.model.kan import KAN

dataset = VideoDataset("data/lr_frames", "data/frames")
loader = DataLoader(dataset, batch_size=1)

dwt = DWT()
vae = VAE()
kan = KAN()

for lr, hr in loader:
    wave = dwt(lr)
    recon, mu, logvar = vae(wave)
    out = kan(recon)

    print("LR:", lr.shape)
    print("Wave:", wave.shape)
    print("After VAE:", recon.shape)
    print("After KAN:", out.shape)
    break