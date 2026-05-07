import sys
import os

# Thêm root project vào path
sys.path.append(os.path.abspath("."))

from src.data.dataset import VideoDataset
from torch.utils.data import DataLoader

dataset = VideoDataset("data/lr_frames", "data/frames")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Total samples:", len(dataset))

for lr, hr in loader:
    print("LR shape:", lr.shape)
    print("HR shape:", hr.shape)
    break