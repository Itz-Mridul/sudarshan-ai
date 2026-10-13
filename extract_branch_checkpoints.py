#!/usr/bin/env python3
"""
extract_branch_checkpoints.py
=============================
Extract and save individual branch checkpoints from an existing fusion model.

THE PROBLEM:
  train_all.py --mode fusion requires branch_a_best.pt, branch_b_best.pt,
  branch_c_best.pt to exist before it can run.
  If you only have fusion_best.pt (the full joint model), this script
  extracts the per-branch weights from it so you can resume with
  --mode joint (fine-tuning) or re-train only the fusion head.

USAGE:
  # Extract branches from the existing fusion checkpoint:
  python extract_branch_checkpoints.py

  # Or specify a custom fusion checkpoint path:
  python extract_branch_checkpoints.py --fusion weights/fusion_best.pt

AFTER RUNNING:
  weights/branch_a_best.pt   <- Branch A weights extracted from fusion
  weights/branch_b_best.pt   <- Branch B weights extracted from fusion
  weights/branch_c_best.pt   <- Branch C weights extracted from fusion
  You can now run: python train_all.py --start_from fusion
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch

sys.path.insert(0, "src")
from model.fusion_model import MultiBranchSteganalyzer
from branches.branch_a_pixel import BranchAClassifier
from branches.branch_b_dct import BranchBClassifier
from branches.branch_c_stats import BranchCClassifier
from utils.checkpoints import extract_state_dict


def filter_prefix(state_dict: dict, prefix: str) -> dict:
    """
    Extract all keys starting with `prefix` and strip the prefix.
    E.g. prefix="branch_a." extracts all Branch A weights.
    """
    result = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            result[new_key] = v
    return result


def save_branch_checkpoint(state_dict: dict, path: str, branch_name: str):
    """Save a branch checkpoint in the v2 metadata format."""
    checkpoint = {
        "state_dict": state_dict,
        "epoch": None,           # unknown (extracted from fusion)
        "val_loss": None,
        "val_acc": None,
        "mode": branch_name,
        "seed": 42,
        "dataset_dir": "data/clean",
        "payload_rate": 0.4,
        "git_commit": "extracted_from_fusion",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "note": "Weights extracted from fusion_best.pt. Use --mode joint to fine-tune.",
    }
    torch.save(checkpoint, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved {branch_name} -> {path} ({size_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract branch checkpoints from fusion model"
    )
    parser.add_argument(
        "--fusion", default="weights/fusion_best.pt",
        help="Path to the fusion model checkpoint"
    )
    parser.add_argument(
        "--out_dir", default="weights",
        help="Directory to save extracted branch checkpoints"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run a quick forward-pass verification on each extracted branch"
    )
    args = parser.parse_args()

    fusion_path = Path(args.fusion)
    if not fusion_path.exists():
        print(f"ERROR: Fusion checkpoint not found: {fusion_path}")
        print("Make sure you have weights/fusion_best.pt before running this script.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load fusion checkpoint ───────────────────────────────────────────────
    print(f"\nLoading fusion checkpoint: {fusion_path}")
    raw = torch.load(fusion_path, map_location="cpu")
    full_state = extract_state_dict(raw)
    print(f"  Total keys in state dict: {len(full_state)}")

    # ── Extract per-branch state dicts ──────────────────────────────────────
    print("\nExtracting branch weights...")
    branch_a_sd = filter_prefix(full_state, "branch_a.")
    branch_b_sd = filter_prefix(full_state, "branch_b.")
    branch_c_sd = filter_prefix(full_state, "branch_c.")

    print(f"  Branch A: {len(branch_a_sd)} parameter tensors")
    print(f"  Branch B: {len(branch_b_sd)} parameter tensors")
    print(f"  Branch C: {len(branch_c_sd)} parameter tensors")

    if not branch_a_sd or not branch_b_sd or not branch_c_sd:
        print("ERROR: Could not extract branch weights. Make sure the fusion model")
        print("  was trained using MultiBranchSteganalyzer (with branch_a, branch_b,")
        print("  branch_c sub-modules). Keys found:")
        print("  ", list(full_state.keys())[:10])
        sys.exit(1)

    # ── Save individual branch checkpoints ──────────────────────────────────
    print("\nSaving branch checkpoints...")

    path_a = os.path.join(args.out_dir, "branch_a_best.pt")
    path_b = os.path.join(args.out_dir, "branch_b_best.pt")
    path_c = os.path.join(args.out_dir, "branch_c_best.pt")

    save_branch_checkpoint(branch_a_sd, path_a, "branch_a")
    save_branch_checkpoint(branch_b_sd, path_b, "branch_b")
    save_branch_checkpoint(branch_c_sd, path_c, "branch_c")

    # ── Verification (optional but recommended) ──────────────────────────────
    if args.verify:
        print("\nVerifying extracted weights via forward pass...")
        dummy_img = torch.zeros(2, 1, 256, 256)

        # Branch A
        try:
            model_a = BranchAClassifier()
            model_a.load_state_dict(branch_a_sd, strict=False)
            model_a.eval()
            with torch.no_grad():
                out = model_a(dummy_img)
            print(f"  Branch A forward pass OK — output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"  Branch A FAILED: {e}")

        # Branch B
        try:
            model_b = BranchBClassifier()
            model_b.load_state_dict(branch_b_sd, strict=False)
            model_b.eval()
            with torch.no_grad():
                out = model_b(dummy_img)
            print(f"  Branch B forward pass OK — output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"  Branch B FAILED: {e}")

        # Branch C
        try:
            model_c = BranchCClassifier()
            model_c.load_state_dict(branch_c_sd, strict=False)
            model_c.eval()
            with torch.no_grad():
                out = model_c(dummy_img)
            print(f"  Branch C forward pass OK — output shape: {tuple(out.shape)}")
        except Exception as e:
            print(f"  Branch C FAILED: {e}")

    # ── Final report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE! Branch checkpoints extracted.")
    print("\nYou can now run the full training pipeline:")
    print("  python train_all.py --start_from fusion    # retrain fusion head only")
    print("  python train_all.py                        # retrain everything from A")
    print("\nOr continue joint fine-tuning:")
    print("  python src/training/train.py --mode joint --lr 1e-5 --epochs 20")
    print("\nOr evaluate the current model:")
    print("  python src/training/evaluate.py --checkpoint weights/fusion_best.pt --mode fusion")
    print("=" * 60)


if __name__ == "__main__":
    main()
