#!/usr/bin/env python3
"""
scripts/generate_training_dataset.py
====================================
Generates a rich, balanced training dataset of realistic paired covers and stego images.
Uses varied procedural textures (gradients, high-frequency edges, smooth regions, Gaussian noise)
and embeds real payloads across both Spatial LSB and Frequency DCT domains.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from data_pipeline.lsb_embedder import embed_lsb
from data_pipeline.dct_embedder import embed_dct


def generate_natural_texture(seed: int, size: int = 256) -> np.ndarray:
    """Generates varied, realistic synthetic covers resembling photographic textures."""
    rng = np.random.default_rng(seed)
    style = seed % 4
    
    if style == 0:
        # Multi-frequency ripple + edge gradient
        x = np.linspace(-3, 3, size)
        y = np.linspace(-3, 3, size)
        xx, yy = np.meshgrid(x, y)
        base = np.sin(xx * (1 + seed % 5)) * np.cos(yy * (1 + seed % 4)) * 50
        base += rng.normal(128, 15, (size, size))
    elif style == 1:
        # Diagonal texture with illumination gradient
        x = np.linspace(0, 200, size)
        gradient = np.tile(x, (size, 1))
        noise = rng.normal(0, 20, (size, size))
        base = gradient + noise
    elif style == 2:
        # High-contrast photographic texture simulation
        low_res = rng.uniform(40, 220, (16, 16))
        img = Image.fromarray(low_res.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
        base = np.array(img, dtype=np.float32) + rng.normal(0, 8, (size, size))
    else:
        # Smooth photographic skin/sky tone with fine sensor noise
        x = np.linspace(80, 180, size)
        y = np.linspace(80, 180, size)
        xx, yy = np.meshgrid(x, y)
        base = (xx + yy) / 2.0 + rng.normal(0, 6, (size, size))

    return np.clip(base, 0, 255).astype(np.uint8)


def create_training_dataset(
    num_pairs: int = 500,
    clean_dir: Path = ROOT_DIR / "data" / "clean",
    stego_dir: Path = ROOT_DIR / "data" / "stego"
):
    clean_dir.mkdir(parents=True, exist_ok=True)
    stego_dir.mkdir(parents=True, exist_ok=True)

    print(f"==============================================================")
    print(f"  Generating {num_pairs} Diverse Paired Images for Training")
    print(f"  Clean Target: {clean_dir}")
    print(f"  Stego Target: {stego_dir}")
    print(f"==============================================================")

    for i in range(1, num_pairs + 1):
        stem = f"carrier_{i:05d}"
        clean_arr = generate_natural_texture(seed=1000 + i)
        
        # Save clean cover
        clean_file = clean_dir / f"{stem}.png"
        Image.fromarray(clean_arr).save(clean_file)

        # Generate payload of varied length (0.2 to 0.8 bpp)
        rng = np.random.default_rng(2000 + i)
        payload_len = int(rng.choice([128, 256, 512, 1024, 2048]))
        payload = rng.bytes(payload_len)

        stego_file = stego_dir / f"{stem}.png"
        if i % 2 == 0:
            # Spatial domain LSB
            stego_arr = embed_lsb(clean_arr, payload, seed=3000 + i)
        else:
            # Frequency domain DCT
            stego_arr = embed_dct(clean_arr, payload, seed=3000 + i, allow_partial=True)

        Image.fromarray(stego_arr).save(stego_file)

        if i % 100 == 0 or i == num_pairs:
            print(f"  [{i:04d}/{num_pairs:04d}] Created {stem}.png (Clean & Stego)")

    print(f"\n✅ Successfully generated {num_pairs} balanced training pairs!\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500)
    args = parser.parse_args()
    create_training_dataset(num_pairs=args.num_pairs)
