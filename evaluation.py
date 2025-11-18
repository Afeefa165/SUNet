#!/usr/bin/env python3
"""
evaluate.py

Usage:
    python evaluate.py --gt_dir /path/to/GT \
                       --noisy_dir /path/to/noisy \
                       --denoised_dir /path/to/denoised \
                       --save_csv results.csv
"""

import argparse
from pathlib import Path
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.color import rgb2gray
import imageio
from natsort import natsorted
from tqdm import tqdm
import csv

def load_img(path):
    img = imageio.imread(path)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.1:
            img = img / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    return img

def safe_ssim(img1, img2):
    g1 = rgb2gray(img1)
    g2 = rgb2gray(img2)
    return compare_ssim(g1, g2, data_range=1.0)

def main(args):
    gt_dir = Path(args.gt_dir)
    noisy_dir = Path(args.noisy_dir)
    denoised_dir = Path(args.denoised_dir)

    # collect files
    gt_files = natsorted([p for ext in ("*.png","*.jpg","*.jpeg","*.bmp") for p in gt_dir.glob(ext)])
    noisy_files = natsorted([p for ext in ("*.png","*.jpg","*.jpeg","*.bmp") for p in noisy_dir.glob(ext)])
    denoised_files = natsorted([p for ext in ("*.png","*.jpg","*.jpeg","*.bmp") for p in denoised_dir.glob(ext)])

    gt_b = {p.stem: p for p in gt_files}
    noisy_b = {p.stem: p for p in noisy_files}
    denoised_b = {p.stem: p for p in denoised_files}

    common = sorted(set(gt_b) & set(noisy_b) & set(denoised_b))

    if len(common) == 0:
        # fallback: match index order
        if len(gt_files) == len(noisy_files) == len(denoised_files):
            iterator = list(range(len(gt_files)))
            use_index = True
        else:
            raise RuntimeError("No matching filenames between GT / noisy / denoised folders.")
    else:
        iterator = common
        use_index = False

    print(f"Evaluating {len(iterator)} images...\n")

    results = []  # for CSV

    noise_psnr_sum = 0
    denoise_psnr_sum = 0
    noise_ssim_sum = 0
    denoise_ssim_sum = 0
    count = 0

    for item in tqdm(iterator, desc="Evaluating", unit="img"):
        if use_index:
            gt_path = gt_files[item]
            noisy_path = noisy_files[item]
            denoised_path = denoised_files[item]
            name = gt_path.name
        else:
            name = item
            gt_path = gt_b[item]
            noisy_path = noisy_b[item]
            denoised_path = denoised_b[item]

        GT = load_img(gt_path)
        Noise = load_img(noisy_path)
        Denoise = load_img(denoised_path)

        # crop to smallest dimensions if mismatch
        h = min(GT.shape[0], Noise.shape[0], Denoise.shape[0])
        w = min(GT.shape[1], Noise.shape[1], Denoise.shape[1])
        GT = GT[:h,:w,:]
        Noise = Noise[:h,:w,:]
        Denoise = Denoise[:h,:w,:]

        n_psnr = compare_psnr(GT, Noise, data_range=1.0)
        d_psnr = compare_psnr(GT, Denoise, data_range=1.0)

        n_ssim = safe_ssim(GT, Noise)
        d_ssim = safe_ssim(GT, Denoise)

        # Accumulate:
        noise_psnr_sum += n_psnr
        denoise_psnr_sum += d_psnr
        noise_ssim_sum += n_ssim
        denoise_ssim_sum += d_ssim
        count += 1

        # Add to result list
        results.append({
            "filename": name,
            "noise_psnr": n_psnr,
            "noise_ssim": n_ssim,
            "denoise_psnr": d_psnr,
            "denoise_ssim": d_ssim
        })

        # Normal print:
        print(f"\nGT: {gt_path.name}")
        print(f"Noise: {noisy_path.name}")
        print(f"Denoise: {denoised_path.name}")
        print(f"  Noise PSNR = {n_psnr:0.4f} dB")
        print(f"  Noise SSIM = {n_ssim:0.4f}")
        print(f"Denoise PSNR* = {d_psnr:0.4f} dB")
        print(f"Denoise SSIM* = {d_ssim:0.4f}")
        print("----------------------------------------------")

    # Final averages:
    print("\n==================== FINISH ====================")
    print(f"Average noise PSNR  = {noise_psnr_sum / count:0.4f} dB")
    print(f"Average noise SSIM  = {noise_ssim_sum / count:0.4f}")
    print(f"Average denoise PSNR = {denoise_psnr_sum / count:0.4f} dB")
    print(f"Average denoise SSIM = {denoise_ssim_sum / count:0.4f}")
    print("================================================")

    # Save CSV if requested
    if args.save_csv:
        csv_path = Path(args.save_csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "noise_psnr", "noise_ssim", "denoise_psnr", "denoise_ssim"]
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        print(f"\nCSV saved to: {csv_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate denoising PSNR/SSIM.")
    parser.add_argument("--gt_dir", required=True, help="Directory of ground truth images")
    parser.add_argument("--noisy_dir", required=True, help="Directory of noisy images")
    parser.add_argument("--denoised_dir", required=True, help="Directory of denoised images")
    parser.add_argument("--save_csv", default=None, help="Path to output CSV file")
    args = parser.parse_args()
    main(args)
              