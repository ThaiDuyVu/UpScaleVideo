import os
import cv2
import random
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=128):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size

        self.samples = []

        for clip in os.listdir(lr_dir):
            lr_clip = os.path.join(lr_dir, clip)
            hr_clip = os.path.join(hr_dir, clip)

            for frame in os.listdir(lr_clip):
                lr_path = os.path.join(lr_clip, frame)
                hr_path = os.path.join(hr_clip, frame)

                if os.path.exists(hr_path):
                    self.samples.append((lr_path, hr_path))

    def __len__(self):
        return len(self.samples)

    def random_crop(self, lr, hr):
        h, w = lr.shape[:2]
        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)

        lr_crop = lr[y:y+self.crop_size, x:x+self.crop_size]
        hr_crop = hr[y*4:(y+self.crop_size)*4, x*4:(x+self.crop_size)*4]

        return lr_crop, hr_crop

    def augment(self, lr, hr):
        if random.random() > 0.5:
            lr = cv2.flip(lr, 1)
            hr = cv2.flip(hr, 1)
        return lr, hr

    def __getitem__(self, idx):
        while True:
            lr_path, hr_path = self.samples[idx]

            lr = cv2.imread(lr_path)
            hr = cv2.imread(hr_path)

            if lr is None or hr is None:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            h, w = lr.shape[:2]

            # 🔥 FIX: bỏ qua ảnh quá nhỏ
            if h < self.crop_size or w < self.crop_size:
                idx = random.randint(0, len(self.samples) - 1)
                continue

            lr, hr = self.random_crop(lr, hr)
            lr, hr = self.augment(lr, hr)

            lr = torch.from_numpy(lr).permute(2,0,1).float() / 255.0
            hr = torch.from_numpy(hr).permute(2,0,1).float() / 255.0

            return lr, hr
