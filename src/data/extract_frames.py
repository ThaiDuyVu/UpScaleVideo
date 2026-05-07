import cv2
import os
from tqdm import tqdm

RAW_DIR = "data/raw_videos"
OUT_DIR = "data/frames"

def extract_frames():
    os.makedirs(OUT_DIR, exist_ok=True)

    for clip_name in os.listdir(RAW_DIR):
        clip_path = os.path.join(RAW_DIR, clip_name, "clip.mp4")
        if not os.path.exists(clip_path):
            continue

        cap = cv2.VideoCapture(clip_path)
        frame_idx = 0

        save_dir = os.path.join(OUT_DIR, clip_name)
        os.makedirs(save_dir, exist_ok=True)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in tqdm(range(total), desc=f"Extracting {clip_name}"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_name = f"frame_{frame_idx:05d}.png"
            cv2.imwrite(os.path.join(save_dir, frame_name), frame)

            frame_idx += 1

        cap.release()

if __name__ == "__main__":
    extract_frames()