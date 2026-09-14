#!/usr/bin/env python3
"""
generate_dct_dataset.py
=======================
Generates frequency-domain steganography (JPEG Jsteg/F5 style) by reading
images from `data/clean` and saving embedded images to `data/stego_dct`.

Output is saved as PNG to preserve exact DCT artifacts without secondary JPEG compression.
"""

import os
import sys
import multiprocessing
import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_pipeline.dct_embedder import create_dct_stego_image


def process_file(args):
    clean_path, out_dir, payload_fraction = args
    stem = os.path.splitext(os.path.basename(clean_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.png")
    
    if os.path.exists(out_path):
        return True # Skip existing
        
    try:
        create_dct_stego_image(clean_path, out_path, payload_fraction)
        return True
    except Exception as e:
        print(f"Failed {clean_path}: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/clean", help="Clean images dir")
    parser.add_argument("--stego", default="data/stego_dct", help="Output stego dir")
    parser.add_argument("--payload", type=float, default=0.1, help="Payload fraction (default 0.1)")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="Parallel workers")
    args = parser.parse_args()

    os.makedirs(args.stego, exist_ok=True)
    
    files = glob.glob(os.path.join(args.clean, "*.*"))
    files = [f for f in files if f.lower().endswith(('.png', '.pgm', '.jpg', '.jpeg', '.bmp'))]
    
    if not files:
        print(f"No images found in {args.clean}")
        return

    print(f"Generating DCT stego dataset from {len(files)} images...")
    print(f"Payload fraction: {args.payload}")
    print(f"Output directory: {args.stego}")
    
    pool_args = [(f, args.stego, args.payload) for f in files]
    
    success = 0
    with multiprocessing.Pool(args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_file, pool_args), total=len(files)):
            if res:
                success += 1
                
    print(f"\nDone! Generated {success}/{len(files)} stego images in {args.stego}")


if __name__ == "__main__":
    main()
