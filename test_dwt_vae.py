from src.data.dataset import VideoDataset
from torch.utils.data import DataLoader
from src.model.wavelet import DWT
from src.model.vae import VAE

dataset = VideoDataset("data/lr_frames", "data/frames")
loader = DataLoader(dataset, batch_size=1)

dwt = DWT()
vae = VAE()

for lr, hr in loader:
    wave = dwt(lr)
    recon, mu, logvar = vae(wave)

    print("LR:", lr.shape)
    print("Wavelet:", wave.shape)
    print("After VAE:", recon.shape)
    break