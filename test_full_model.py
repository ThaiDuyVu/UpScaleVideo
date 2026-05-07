from src.data.dataset import VideoDataset
from torch.utils.data import DataLoader

from src.model.generator import Generator

dataset = VideoDataset("data/lr_frames", "data/frames")
loader = DataLoader(dataset, batch_size=1)

model = Generator()

for lr, hr in loader:
    out, mu, logvar = model(lr)

    print("LR:", lr.shape)
    print("Output:", out.shape)
    break