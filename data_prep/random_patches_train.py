#!/usr/bin/env python3
"""
Random-sigma patch extractor + gaussian noise
Equivalent to your first MATLAB block.
"""
import os
import argparse
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import random

def add_gaussian_noise(img_patch, sigma):
    # img_patch expected uint8 HxWxC
    var = (sigma / 256.0) ** 2
    sigma_pixel = np.sqrt(var) * 255.0
    noise = np.random.normal(0, sigma_pixel, img_patch.shape)
    noisy = img_patch.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def main(src_dir, out_target_dir, out_input_dir, patch_size=256, patches_per_image=100, seed=1234):
    os.makedirs(out_target_dir, exist_ok=True)
    os.makedirs(out_input_dir, exist_ok=True)
    files = natsorted(glob(os.path.join(src_dir, '*.png')) + glob(os.path.join(src_dir, '*.jpg')))
    count = 0
    random.seed(seed)
    for file in tqdm(files, desc='Images'):
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        H, W = img.shape[:2]
        if H < patch_size or W < patch_size:
            # skip or pad as you prefer
            continue
        for _ in range(patches_per_image):
            y = random.randint(0, H - patch_size)
            x = random.randint(0, W - patch_size)
            C = img[y:y+patch_size, x:x+patch_size, :]
            sig = random.randint(5, 50)  # same distribution
            noisy = add_gaussian_noise(C, sig)
            count += 1
            cv2.imwrite(os.path.join(out_target_dir, f"{count}.png"), C)
            cv2.imwrite(os.path.join(out_input_dir, f"{count}.png"), noisy)
    print("Finished. Total patches:", count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="Path to HR images (source)")
    parser.add_argument("--out_target_dir", required=True, help="Where to save clean patches (target)")
    parser.add_argument("--out_input_dir", required=True, help="Where to save noisy patches (input)")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--patches_per_image", type=int, default=100)
    args = parser.parse_args()
    main(args.src_dir, args.out_target_dir, args.out_input_dir, args.patch_size, args.patches_per_image)
