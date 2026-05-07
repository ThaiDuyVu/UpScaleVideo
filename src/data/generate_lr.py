import cv2
import os
import numpy as np
from tqdm import tqdm

HR_DIR = "data/frames"
LR_DIR = "data/lr_frames"

SCALE = 4

def add_gaussian_noise(img, sigma=2):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def generate_lr():
    os.makedirs(LR_DIR, exist_ok=True)

    for clip_name in os.listdir(HR_DIR):
        hr_clip_dir = os.path.join(HR_DIR, clip_name)
        lr_clip_dir = os.path.join(LR_DIR, clip_name)
        os.makedirs(lr_clip_dir, exist_ok=True)

        for frame_name in tqdm(os.listdir(hr_clip_dir), desc=f"LR {clip_name}"):
            hr_path = os.path.join(hr_clip_dir, frame_name)
            hr = cv2.imread(hr_path)

            h, w = hr.shape[:2]

            # Downscale
            lr = cv2.resize(hr, (w//SCALE, h//SCALE), interpolation=cv2.INTER_CUBIC)

            # Optional noise
            lr = add_gaussian_noise(lr)

            cv2.imwrite(os.path.join(lr_clip_dir, frame_name), lr)

if __name__ == "__main__":
    generate_lr()