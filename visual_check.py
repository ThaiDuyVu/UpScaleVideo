import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data.dataset import VideoDataset

# Load dataset
dataset = VideoDataset("data/lr_frames", "data/frames")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Lấy 1 sample
for lr, hr in loader:
    break

# Convert tensor → numpy image
lr_img = lr[0].permute(1, 2, 0).numpy()
hr_img = hr[0].permute(1, 2, 0).numpy()

# Hiển thị
plt.subplot(1, 2, 1)
plt.title("LR")
plt.imshow(lr_img)

plt.subplot(1, 2, 2)
plt.title("HR")
plt.imshow(hr_img)

plt.show()