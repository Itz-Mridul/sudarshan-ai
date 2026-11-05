#!/usr/bin/env python3
"""
scripts/evaluate_test.py
========================
Evaluates the trained Sudarshana Steganalyzer on the held-out test split.
Outputs clean accuracy, stego accuracy, and overall classification score.
"""

import os
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from model.fusion_model import MultiBranchSteganalyzer
from data_pipeline.dataset import get_dataloaders
from utils.checkpoints import extract_state_dict


def run_evaluation(
    weights_path: str = "weights/fusion_best.pt",
    clean_dir: str = "data/clean",
    stego_dir: str = "data/stego",
    batch_size: int = 32,
    max_images: int = None
):
    print("=" * 65)
    print("  🛡️ Sudarshana Model Evaluation (Held-Out Test Set)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_file = ROOT_DIR / weights_path
    if not ckpt_file.exists():
        print(f"❌ Error: Weights file not found at '{ckpt_file}'")
        return

    # Load dataloaders
    try:
        _, _, test_loader = get_dataloaders(
            clean_dir=str(ROOT_DIR / clean_dir),
            stego_dir=str(ROOT_DIR / stego_dir),
            batch_size=batch_size,
            max_images=max_images
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Initialize model
    model = MultiBranchSteganalyzer().to(device)
    raw = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(extract_state_dict(raw), strict=False)
    model.eval()

    total_clean_correct = 0
    total_clean_samples = 0
    total_stego_correct = 0
    total_stego_samples = 0

    print("Evaluating test batches...")
    with torch.no_grad():
        for clean_imgs, stego_imgs in test_loader:
            B = clean_imgs.size(0)
            images = torch.cat([clean_imgs, stego_imgs], dim=0).to(device)
            labels = torch.cat([
                torch.zeros(B, dtype=torch.long),
                torch.ones(B, dtype=torch.long)
            ], dim=0).to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            # Separate clean (first B) and stego (second B)
            clean_preds = preds[:B]
            stego_preds = preds[B:]

            total_clean_correct += (clean_preds == 0).sum().item()
            total_clean_samples += B
            total_stego_correct += (stego_preds == 1).sum().item()
            total_stego_samples += B

    total_correct = total_clean_correct + total_stego_correct
    total_samples = total_clean_samples + total_stego_samples

    clean_acc = (total_clean_correct / total_clean_samples * 100) if total_clean_samples > 0 else 0
    stego_acc = (total_stego_correct / total_stego_samples * 100) if total_stego_samples > 0 else 0
    overall_acc = (total_correct / total_samples * 100) if total_samples > 0 else 0

    print("\n" + "-" * 65)
    print(f"  Clean Cover Accuracy  (TN rate): {clean_acc:6.2f}% ({total_clean_correct}/{total_clean_samples})")
    print(f"  Stego Carrier Accuracy (TP rate): {stego_acc:6.2f}% ({total_stego_correct}/{total_stego_samples})")
    print(f"  Overall Test Accuracy           : {overall_acc:6.2f}% ({total_correct}/{total_samples})")
    print("-" * 65)
    print(f"🏆 Final Verdict: Model successfully evaluated on {total_samples} held-out images.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Sudarshana Model")
    parser.add_argument("--weights", type=str, default="weights/fusion_best.pt", help="Path to checkpoint")
    parser.add_argument("--max_images", type=int, default=None, help="Limit images evaluated")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_evaluation(weights_path=args.weights, max_images=args.max_images, batch_size=args.batch_size)
