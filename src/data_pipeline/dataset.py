"""
dataset.py
==========
PyTorch Dataset for Steganography Detection

WHAT THIS DOES:
  Wraps the clean/ and stego/ image folders into a proper PyTorch Dataset
  that the training loop can use with a DataLoader.

STRUCTURE EXPECTED:
  data/
  ├── clean/    ← label 0 (no hidden data)
  └── stego/    ← label 1 (hidden data present), SAME filenames as clean/

CRITICAL DESIGN DECISIONS (v2 — bug-fixed):

  Decision 1 — Pair-safe train/val split:
    The split is done on SOURCE IMAGE NAMES, not on individual samples.
    If "img_001.png" is in the validation set, BOTH data/clean/img_001.png
    AND data/stego/img_001.png end up in validation.
    This is mandatory because a clean/stego pair carries shared scene content.
    If they land in different splits, the model can learn to distinguish
    training scenes from validation scenes — inflating accuracy by 15-20%.

  Decision 2 — Lossless preprocessing (NO bilinear resize for LSB):
    Bilinear interpolation averages neighbouring pixel values.
    This destroys the 1-bit LSB payload:
      pixel 200 (clean) and 201 (stego) both become 200.5 → rounded to 200.
      The label becomes meaningless.
    SOLUTION: Random crops for training, centre crops for validation.
    This preserves exact pixel values. Images must be at least crop_size×crop_size.
    For 512×512 BOSS images with crop_size=256 — works perfectly.
    For smaller images: use the full image (no crop) and reject via assert.

STUDENT A TASK: This is used in Weeks 3-4 once your data is ready.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Set

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SteganalysisDataset(Dataset):
    """
    Loads clean and stego images from two folders.
    Returns (image_tensor, label) pairs.

    label = 0 → clean image (no hidden data)
    label = 1 → stego image (hidden data present)

    IMPORTANT: stego/ filenames must exactly match clean/ filenames.
    The pair-safe split in get_dataloaders() depends on this.
    """

    def __init__(
        self,
        clean_dir: str,
        stego_dir: str,
        allowed_stems: Optional[Set[str]] = None,
        transform=None,
    ):
        """
        Args:
            clean_dir    : path to folder with clean images (label=0)
            stego_dir    : path to folder with stego images (label=1)
            allowed_stems: set of file stems (without extension) to include.
                           If None, all files are included.
                           Pass train_stems or val_stems from the split helper.
            transform    : torchvision transforms to apply to each image
        """
        self.transform = transform
        self.samples: List[Tuple[str, str]] = []   # (clean_path, stego_path)

        extensions = {".png", ".pgm", ".bmp"}
        # Note: .jpg / .jpeg intentionally excluded — JPEG re-saves will
        # corrupt LSB payload (see lsb_embedder.py). Clean inputs may be
        # JPEG; stego outputs are always saved as PNG.

        clean_path = Path(clean_dir)
        stego_path = Path(stego_dir)

        # ── Filename-keyed matching (v2 — fixes silent mispair bug) ──────────
        # Build stem → Path dicts so pairs are matched by NAME, not position.
        # If a file is missing in one folder, it is skipped with a warning
        # instead of silently pairing with the wrong image.
        clean_map = {
            f.stem: f for f in clean_path.iterdir()
            if f.suffix.lower() in extensions
        }
        stego_map = {
            f.stem: f for f in stego_path.iterdir()
            if f.suffix.lower() in extensions
        }

        all_stems = sorted(clean_map.keys() & stego_map.keys())

        if allowed_stems is not None:
            all_stems = [s for s in all_stems if s in allowed_stems]

        # Report stems present in only one folder
        only_clean = clean_map.keys() - stego_map.keys()
        only_stego = stego_map.keys() - clean_map.keys()
        if only_clean:
            print(f"[WARNING] {len(only_clean)} clean images have no stego counterpart — skipped.")
        if only_stego:
            print(f"[WARNING] {len(only_stego)} stego images have no clean counterpart — skipped.")

        for stem in all_stems:
            self.samples.append((str(clean_map[stem]), str(stego_map[stem])))

        n = len(self.samples)
        if n == 0:
            raise RuntimeError(
                f"No valid images found in:\n"
                f"  clean: {clean_dir}\n"
                f"  stego: {stego_dir}\n"
                "Accepted formats: PNG, PGM, BMP (JPEG excluded — LSB is lossless only).\n"
                "Run: bash scripts/setup_boss_dataset.sh  to populate the dataset."
            )
        else:
            print(f"Dataset: {n} clean + {n} stego = {2 * n} total images"
                  + (f" [from {len(allowed_stems)} source stems]" if allowed_stems else ""))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        clean_path, stego_path = self.samples[idx]

        # Load as grayscale — SRM filters operate on grayscale only
        clean_img = Image.open(clean_path).convert("L")
        stego_img = Image.open(stego_path).convert("L")

        # To ensure paired random cropping and flipping, we seed the RNG identical for both
        seed = np.random.randint(2147483647)
        
        if self.transform is not None:
            import torch
            torch.manual_seed(seed)
            clean_tensor = self.transform(clean_img)
            torch.manual_seed(seed)
            stego_tensor = self.transform(stego_img)
        else:
            clean_tensor = T.ToTensor()(clean_img)
            stego_tensor = T.ToTensor()(stego_img)

        return clean_tensor, stego_tensor


# ─── Lossless Transforms ──────────────────────────────────────────────────────

def get_transforms(crop_size: int = 256):
    """
    Returns (train_transform, val_transform).

    NO RESIZE — uses random/centre crops instead to preserve pixel values.

    For LSB detection, even a single pixel value rounding from interpolation
    can flip a true positive to a false negative. Crops guarantee that every
    pixel reaching the model is exactly the value set by embed_lsb().

    Augmentation policy:
      - RandomCrop: shifts which pixels are seen, does NOT alter pixel values.
      - RandomHorizontalFlip: LSB is independent of spatial position → safe.
      - NO colour jitter, rotation, or scaling — these all alter pixel values.

    Args:
        crop_size: size of the square crop (default 256).
                   Images must be at least crop_size × crop_size.
                   BOSS dataset (512×512) works perfectly.
    """
    # Validation: deterministic centre crop
    val_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])      # scale to [-1, 1]
    ])

    # Training: random crop + horizontal flip
    train_transform = T.Compose([
        T.RandomCrop(crop_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, val_transform


# ─── Pair-Safe Split Helper ───────────────────────────────────────────────────

def split_stems_three_way(
    clean_dir: str,
    stego_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_images: Optional[int] = None,
    seed: int = 42
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split image stems into TRAIN / VAL / TEST sets (default 70/15/15).
    
    Splitting is done on source stems BEFORE dataset creation so a
    clean/stego pair is NEVER split across two sets.
    """
    assert val_split + test_split < 1.0, "Splits must sum to < 1.0"

    extensions = {".png", ".pgm", ".bmp"}
    clean_path = Path(clean_dir)
    stego_path = Path(stego_dir)

    clean_stems = {f.stem for f in clean_path.iterdir() if f.suffix.lower() in extensions}
    stego_stems = {f.stem for f in stego_path.iterdir() if f.suffix.lower() in extensions}
    
    # Only use stems that exist in BOTH directories (Fix #3)
    valid_stems = clean_stems & stego_stems
    all_stems = sorted(list(valid_stems))

    rng = np.random.default_rng(seed)
    rng.shuffle(all_stems)

    if max_images is not None:
        all_stems = all_stems[:max_images]

    n_total = len(all_stems)
    n_test = max(1, int(n_total * test_split))
    n_val  = max(1, int(n_total * val_split))

    test_stems  = set(all_stems[:n_test])
    val_stems   = set(all_stems[n_test:n_test + n_val])
    train_stems = set(all_stems[n_test + n_val:])

    train_pct = len(train_stems) / n_total * 100
    val_pct   = len(val_stems)   / n_total * 100
    test_pct  = len(test_stems)  / n_total * 100
    print(f"Pair-safe split: {len(train_stems)} train ({train_pct:.0f}%), "
          f"{len(val_stems)} val ({val_pct:.0f}%), "
          f"{len(test_stems)} test ({test_pct:.0f}%)")

    return train_stems, val_stems, test_stems


# Keep the old 2-way split as an alias for backward compatibility
def split_stems_by_pair(
    clean_dir: str,
    stego_dir: str = None,
    val_split: float = 0.20,
    max_images: Optional[int] = None,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Split stems into train/val only (backward compat).
    Prefer split_stems_three_way() for new code.
    """
    # Derive stego_dir from clean_dir if not explicitly provided (backward compat)
    if stego_dir is None:
        stego_dir = clean_dir.replace("clean", "stego")
    
    # Use a tiny test_split so the assert passes; merge test back into train
    train, val, test = split_stems_three_way(
        clean_dir, stego_dir, val_split=val_split, test_split=0.01,
        max_images=max_images, seed=seed
    )
    return train | test, val


# ─── Flat (image, label) Dataset — for baselines (Xu-Net, etc.) ─────────────

class FlatDataset(torch.utils.data.Dataset):
    """
    Standard (image_tensor, label) dataset for use with Xu-Net and other
    baselines that expect a flat per-sample format instead of paired batches.

    label 0 = clean, label 1 = stego
    """

    def __init__(
        self,
        clean_dir: str,
        stego_dir: str,
        allowed_stems: Optional[Set[str]] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []  # (path, label)

        extensions = {".png", ".pgm", ".bmp"}
        clean_path = Path(clean_dir)
        stego_path = Path(stego_dir)

        clean_map = {f.stem: f for f in clean_path.iterdir() if f.suffix.lower() in extensions}
        stego_map = {f.stem: f for f in stego_path.iterdir() if f.suffix.lower() in extensions}
        stems = sorted(clean_map.keys() & stego_map.keys())
        if allowed_stems is not None:
            stems = [s for s in stems if s in allowed_stems]

        for stem in stems:
            self.samples.append((str(clean_map[stem]), 0))
            self.samples.append((str(stego_map[stem]), 1))

        print(f"FlatDataset: {len(stems)} pairs → {len(self.samples)} samples (50/50 balance)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)
        return img, label


# ─── DataLoader Builder ───────────────────────────────────────────────────────

def get_dataloaders(
    clean_dir: str,
    stego_dir: str,
    batch_size: int = 32,
    crop_size: int = 256,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_images: Optional[int] = None,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders with a PAIR-SAFE 3-way split.

    Default: 70% train, 15% val, 15% test.
    The TEST loader must ONLY be used for final evaluation — never for
    hyperparameter tuning or threshold selection.

    Args:
        clean_dir  : folder with clean images
        stego_dir  : folder with stego images (same filenames as clean)
        batch_size : images per batch
        crop_size  : crop size (NOT resize — preserves LSB pixel values)
        val_split  : fraction held out for validation
        test_split : fraction held out for final test
        max_images : limit total pairs considered (for quick experiments)
        num_workers: parallel data loading workers
        seed       : random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader) tuple
    """
    train_transform, val_transform = get_transforms(crop_size)

    # Step 1: 3-way split on stems (pair-safe)
    train_stems, val_stems, test_stems = split_stems_three_way(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        val_split=val_split,
        test_split=test_split,
        max_images=max_images,
        seed=seed
    )

    # Step 2: Create SEPARATE Dataset objects for each split
    train_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        allowed_stems=train_stems,
        transform=train_transform,
    )
    val_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        allowed_stems=val_stems,
        transform=val_transform,
    )
    test_dataset = SteganalysisDataset(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        allowed_stems=test_stems,
        transform=val_transform,   # no augmentation on test
    )

    # pin_memory speeds up CPU→GPU transfers but is NOT supported on MPS
    import torch
    use_pin = torch.cuda.is_available()  # True only for NVIDIA GPU
    
    # Fix #4: Disable drop_last if train_dataset is too small
    drop_last = len(train_dataset) >= batch_size

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin,
        drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin
    )

    print(f"Train batches: {len(train_loader)}, "
          f"Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")
    return train_loader, val_loader, test_loader


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", default="data/clean", help="Clean images directory")
    parser.add_argument("--stego", default="data/stego", help="Stego images directory")
    parser.add_argument("--verify", action="store_true", help="Print sample info")
    parser.add_argument("--samples", type=int, default=5, help="Samples to print")
    args = parser.parse_args()

    _, val_tf = get_transforms()
    dataset = SteganalysisDataset(args.clean, args.stego, transform=val_tf)

    print(f"\nTotal image pairs in dataset: {len(dataset)}")

    if args.verify:
        print(f"\nFirst {args.samples} samples:")
        for i in range(min(args.samples, len(dataset))):
            clean, stego = dataset[i]
            print(f"[{i}] clean={tuple(clean.shape)}, stego={tuple(stego.shape)}, "
                  f"changed_pixels={(clean != stego).any().item()}")
