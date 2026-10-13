#!/usr/bin/env python3
"""
setup_alaska2.py
================
ALASKA2 Dataset Download and Setup for Cross-Dataset Evaluation

WHAT THIS DOES:
  ALASKA2 is a large-scale steganography benchmark with 80,000 images
  from multiple cameras. It was used in the Kaggle ALASKA2 competition
  and is the gold standard for testing cross-dataset generalization.

WHY WE NEED IT:
  Our model is trained on BOSSbase (10,000 grayscale images from one camera).
  The IEEE plan requires us to test on a DIFFERENT dataset to prove
  generalization. If the model only works on BOSSbase, it is useless
  for defense. ALASKA2 has:
    - Different cameras (Sony, Canon, Nikon, etc.)
    - Color JPEG images (not grayscale PGM)
    - Different resolutions (512×512)
    - Three stego algorithms: JMiPOD, J-UNIWARD, UERD

USAGE:
  # Download and setup
  python scripts/setup_alaska2.py --output data/alaska2

  # Quick test with subset
  python scripts/setup_alaska2.py --output data/alaska2 --max_images 500
"""

import os
import sys
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Tuple, Optional
import shutil

import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def discover_images(directory: str, extensions: set = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}) -> List[str]:
    """Find all image files in a directory recursively."""
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(root, f))
    images.sort()
    return images


def create_pair_safe_split(
    image_names: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split image names into train/val/test ensuring that ALL variants
    of a source image stay in the same split (preventing data leakage).

    This is CRITICAL: if clean_001.jpg is in train, then stego_001.jpg
    must also be in train. Otherwise, the model learns scene identity
    instead of steganographic artifacts.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Extract unique base names (strip path, extension, and stego prefix)
    unique_bases = list(set(
        Path(name).stem.replace("stego_", "").replace("clean_", "")
        for name in image_names
    ))
    random.shuffle(unique_bases)

    n = len(unique_bases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_bases = set(unique_bases[:n_train])
    val_bases = set(unique_bases[n_train:n_train + n_val])
    test_bases = set(unique_bases[n_train + n_val:])

    return list(train_bases), list(val_bases), list(test_bases)


def setup_alaska2_from_kaggle(output_dir: str, max_images: Optional[int] = None):
    """
    Setup ALASKA2 dataset structure for cross-dataset evaluation.

    Expected download structure (from Kaggle):
      alaska2/
      ├── Cover/         ← Clean images (75,000 JPEGs)
      ├── JMiPOD/        ← Stego variant 1
      ├── JUNIWARD/      ← Stego variant 2
      └── UERD/          ← Stego variant 3
    """
    cover_dir = os.path.join(output_dir, "Cover")

    if not os.path.exists(cover_dir):
        print("=" * 60)
        print("  ALASKA2 Dataset Setup")
        print("=" * 60)
        print()
        print("  The ALASKA2 dataset must be downloaded from Kaggle:")
        print()
        print("  Option 1: Kaggle CLI (recommended)")
        print("    pip install kaggle")
        print("    kaggle competitions download -c alaska2-image-steganalysis")
        print(f"    unzip alaska2-image-steganalysis.zip -d {output_dir}")
        print()
        print("  Option 2: Manual download")
        print("    1. Go to: https://www.kaggle.com/c/alaska2-image-steganalysis/data")
        print("    2. Download the competition data")
        print(f"    3. Extract to: {output_dir}/")
        print()
        print("  Expected structure after extraction:")
        print(f"    {output_dir}/")
        print(f"    ├── Cover/       ← Clean images")
        print(f"    ├── JMiPOD/      ← Stego method 1")
        print(f"    ├── JUNIWARD/    ← Stego method 2")
        print(f"    └── UERD/        ← Stego method 3")
        print()

        # Create directory structure for manual placement
        os.makedirs(os.path.join(output_dir, "Cover"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "JMiPOD"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "JUNIWARD"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "UERD"), exist_ok=True)

        print(f"  Created directory structure at: {output_dir}")
        print("  Place the images in the correct folders and re-run this script.")
        return

    # If data exists, organize it
    print("=" * 60)
    print("  ALASKA2 Dataset Found — Organizing...")
    print("=" * 60)

    # Count images
    cover_images = discover_images(cover_dir)
    stego_dirs = {}
    for method in ["JMiPOD", "JUNIWARD", "UERD"]:
        method_dir = os.path.join(output_dir, method)
        if os.path.exists(method_dir):
            stego_dirs[method] = discover_images(method_dir)

    if max_images:
        cover_images = cover_images[:max_images]

    print(f"  Cover images:    {len(cover_images)}")
    for method, imgs in stego_dirs.items():
        print(f"  {method} images: {len(imgs)}")

    # Create pair-safe split
    base_names = [Path(img).stem for img in cover_images]
    train_names, val_names, test_names = create_pair_safe_split(base_names)

    print()
    print(f"  Train split: {len(train_names)} source images")
    print(f"  Val split:   {len(val_names)} source images")
    print(f"  Test split:  {len(test_names)} source images")

    # Write split files
    splits_dir = os.path.join(output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    for split_name, names in [("train", train_names), ("val", val_names), ("test", test_names)]:
        with open(os.path.join(splits_dir, f"{split_name}.txt"), "w") as f:
            for name in sorted(names):
                f.write(name + "\n")

    print(f"\n  Split files saved to: {splits_dir}/")
    print("  ✅ ALASKA2 dataset ready for cross-dataset evaluation!")

    # Generate dataset info
    info_path = os.path.join(output_dir, "DATASET_INFO.md")
    with open(info_path, "w") as f:
        f.write("# ALASKA2 Dataset Info\n\n")
        f.write(f"- **Cover images:** {len(cover_images)}\n")
        for method, imgs in stego_dirs.items():
            f.write(f"- **{method} stego:** {len(imgs)}\n")
        f.write(f"\n## Splits\n")
        f.write(f"- Train: {len(train_names)} source images\n")
        f.write(f"- Validation: {len(val_names)} source images\n")
        f.write(f"- Test: {len(test_names)} source images\n")
        f.write(f"\n## Leakage Prevention\n")
        f.write("All variants (Cover + JMiPOD + JUNIWARD + UERD) of the same source\n")
        f.write("image are guaranteed to be in the SAME split.\n")

    print(f"  Dataset info saved to: {info_path}")


class ALASKA2Dataset:
    """
    PyTorch-compatible loader for ALASKA2 cross-dataset evaluation.

    This creates DataLoaders that pair Cover images with their stego
    variants for fair evaluation against the model trained on BOSSbase.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        stego_method: str = "JUNIWARD",
        crop_size: int = 256,
    ):
        """
        Args:
            root_dir: Path to ALASKA2 dataset root
            split: "train", "val", or "test"
            stego_method: "JMiPOD", "JUNIWARD", or "UERD"
            crop_size: Size of crops (must match training crop size)
        """
        self.root_dir = root_dir
        self.stego_method = stego_method
        self.crop_size = crop_size

        # Load split file
        split_file = os.path.join(root_dir, "splits", f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run: python scripts/setup_alaska2.py --output {root_dir}"
            )

        with open(split_file, "r") as f:
            self.image_names = [line.strip() for line in f if line.strip()]

        # Build pairs: (cover_path, stego_path)
        self.pairs = []
        cover_dir = os.path.join(root_dir, "Cover")
        stego_dir = os.path.join(root_dir, stego_method)

        for name in self.image_names:
            # Try common extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                cover_path = os.path.join(cover_dir, name + ext)
                stego_path = os.path.join(stego_dir, name + ext)
                if os.path.exists(cover_path) and os.path.exists(stego_path):
                    self.pairs.append((cover_path, stego_path))
                    break

    def get_dataloader(self, batch_size: int = 16, num_workers: int = 2):
        """Create a DataLoader for evaluation."""
        from torch.utils.data import Dataset as TorchDataset, DataLoader
        import torchvision.transforms.functional as TF

        outer = self

        class _Dataset(TorchDataset):
            def __init__(self):
                self.samples = []
                for cover_path, stego_path in outer.pairs:
                    self.samples.append((cover_path, 0))  # Clean
                    self.samples.append((stego_path, 1))  # Stego

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                path, label = self.samples[idx]
                img = Image.open(path).convert("L")

                # Centre crop to match training
                w, h = img.size
                cs = outer.crop_size
                if w >= cs and h >= cs:
                    left = (w - cs) // 2
                    top = (h - cs) // 2
                    img = img.crop((left, top, left + cs, top + cs))

                arr = np.array(img, dtype=np.float32)
                arr = (arr / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
                tensor = torch.tensor(arr).unsqueeze(0)  # (1, H, W)
                return tensor, label

        dataset = _Dataset()
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALASKA2 Dataset Setup")
    parser.add_argument("--output", type=str, default="data/alaska2",
                        help="Output directory for ALASKA2 dataset")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of cover images to use (for quick testing)")
    args = parser.parse_args()

    setup_alaska2_from_kaggle(args.output, args.max_images)
