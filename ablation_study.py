"""
ablation_study.py
=================
8-Way Ablation Study for Sudarshana / StegShield

WHAT THIS DOES:
  Systematically tests the model with different branch combinations to prove
  that each branch contributes meaningfully to the final accuracy.

  This is MANDATORY for the IEEE paper — reviewers will reject the paper
  if you claim "multi-branch is better" without proving it with ablation.

WHY THIS MATTERS FOR DRDO/ISRO:
  Defense scientists will ask: "How do you know the frequency branch actually
  helps? Maybe only the pixel branch matters." This script generates the exact
  numbers to answer that question.

ABLATION CONFIGURATIONS:
  1. Full Model (A + B + C)         — the complete Sudarshana system
  2. Branch A only (Pixel CNN)      — spatial-only baseline
  3. Branch B only (DCT CNN)        — frequency-only baseline
  4. Branch C only (Stats MLP)      — statistics-only baseline
  5. A + B (no Stats)               — pixel + frequency, no statistics
  6. A + C (no DCT)                 — pixel + statistics, no frequency
  7. B + C (no Pixel)               — frequency + statistics, no pixel
  8. Dense Fusion vs Cross-Attention — architecture comparison

USAGE:
  python ablation_study.py --data_dir data/stego_dataset --epochs 20
  python ablation_study.py --weights weights/fusion_best.pt --eval_only
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from model.fusion_model import MultiBranchSteganalyzer


class AblatedModel(nn.Module):
    """
    A wrapper around MultiBranchSteganalyzer that can disable specific branches
    by zeroing out their feature vectors during the forward pass.

    This allows fair comparison: the model architecture stays the same,
    but specific branches are "silenced" to measure their contribution.
    """

    def __init__(
        self,
        base_model: MultiBranchSteganalyzer,
        disable_a: bool = False,
        disable_b: bool = False,
        disable_c: bool = False,
        disable_d: bool = False,
    ):
        super().__init__()
        self.model = base_model
        self.disable_a = disable_a
        self.disable_b = disable_b
        self.disable_c = disable_c
        self.disable_d = disable_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get branch features using the base model's internal methods
        features_out = self.model(x, return_branch_features=True)
        logits = features_out[0]
        feat_a = features_out[1]
        feat_b = features_out[2]
        feat_c = features_out[3]
        feat_d = features_out[4] if len(features_out) > 4 else None

        # Zero out disabled branches
        if self.disable_a:
            feat_a = torch.zeros_like(feat_a)
        if self.disable_b:
            feat_b = torch.zeros_like(feat_b)
        if self.disable_c:
            feat_c = torch.zeros_like(feat_c)
        if self.disable_d and feat_d is not None:
            feat_d = torch.zeros_like(feat_d)

        # Re-normalize
        feat_a_norm = self.model.norm_a(feat_a)
        feat_b_norm = self.model.norm_b(feat_b)
        feat_c_norm = self.model.norm_c(feat_c)
        feat_d_norm = self.model.norm_d(feat_d) if feat_d is not None else None

        # Re-fuse
        if self.model.fusion_type == "cross_attention":
            logits = self.model.fusion(feat_a_norm, feat_b_norm, feat_c_norm)
        else:
            if feat_d_norm is not None:
                combined = torch.cat([feat_a_norm, feat_b_norm, feat_c_norm, feat_d_norm], dim=1)
            else:
                combined = torch.cat([feat_a_norm, feat_b_norm, feat_c_norm], dim=1)
            logits = self.model.fusion(combined)

        return logits


# ─── Ablation Configurations ─────────────────────────────────────────────────

ABLATION_CONFIGS = {
    "Full Model (A+B+C+D)":     {"disable_a": False, "disable_b": False, "disable_c": False, "disable_d": False},
    "Branch A only (Pixel)":    {"disable_a": False, "disable_b": True,  "disable_c": True,  "disable_d": True},
    "Branch B only (DCT)":      {"disable_a": True,  "disable_b": False, "disable_c": True,  "disable_d": True},
    "Branch C only (Stats)":    {"disable_a": True,  "disable_b": True,  "disable_c": False, "disable_d": True},
    "Branch D only (GenAI)":    {"disable_a": True,  "disable_b": True,  "disable_c": True,  "disable_d": False},
    "A+B (no Stats/Gen)":       {"disable_a": False, "disable_b": False, "disable_c": True,  "disable_d": True},
    "A+B+C (Tri-Branch)":       {"disable_a": False, "disable_b": False, "disable_c": False, "disable_d": True},
    "B+C+D (no Pixel)":         {"disable_a": True,  "disable_b": False, "disable_c": False, "disable_d": False},
}


def evaluate_ablation(model, dataloader, device):
    """
    Evaluate a model on a dataset and return accuracy, precision, recall, F1.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)

    # For binary classification (0=clean, 1=stego)
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def run_ablation_study(model_path: str, dataloader, device):
    """
    Run the full 8-way ablation study and print results in a table.
    """
    print("=" * 80)
    print("  SUDARSHANA — 8-Way Ablation Study")
    print(f"  Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    print("=" * 80)
    print()

    results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        print(f"  Testing: {config_name} ...")

        # Load fresh model
        model = MultiBranchSteganalyzer(fusion_type="dense")
        if os.path.exists(model_path):
            raw = torch.load(model_path, map_location=device)
            state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            model.load_state_dict(state, strict=False)

        # Create ablated version
        ablated = AblatedModel(model, **config).to(device)

        # Evaluate
        metrics = evaluate_ablation(ablated, dataloader, device)
        results[config_name] = metrics

        print(f"    → Accuracy: {metrics['accuracy']*100:.1f}%  |  "
              f"F1: {metrics['f1']*100:.1f}%  |  "
              f"Precision: {metrics['precision']*100:.1f}%  |  "
              f"Recall: {metrics['recall']*100:.1f}%")

    # Also test Cross-Attention fusion
    print(f"\n  Testing: Cross-Attention Fusion (A+B+C) ...")
    model_ca = MultiBranchSteganalyzer(fusion_type="cross_attention")
    if os.path.exists(model_path):
        # Note: Cross-attention model has different fusion weights,
        # so we load with strict=False
        raw = torch.load(model_path, map_location=device)
        state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
        model_ca.load_state_dict(state, strict=False)
    model_ca = model_ca.to(device)
    metrics_ca = evaluate_ablation(model_ca, dataloader, device)
    results["Cross-Attention (A+B+C)"] = metrics_ca
    print(f"    → Accuracy: {metrics_ca['accuracy']*100:.1f}%  |  "
          f"F1: {metrics_ca['f1']*100:.1f}%  |  "
          f"Precision: {metrics_ca['precision']*100:.1f}%  |  "
          f"Recall: {metrics_ca['recall']*100:.1f}%")

    # Print final table
    print("\n")
    print("=" * 80)
    print("  ABLATION RESULTS TABLE (for IEEE Paper)")
    print("=" * 80)
    print(f"  {'Configuration':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 70)

    for name, m in results.items():
        print(f"  {name:<30} {m['accuracy']*100:>9.1f}% {m['precision']*100:>9.1f}% "
              f"{m['recall']*100:>9.1f}% {m['f1']*100:>9.1f}%")

    print("  " + "-" * 70)
    print()

    # Identify the best configuration
    best_config = max(results, key=lambda k: results[k]["f1"])
    print(f"  Best configuration: {best_config} (F1={results[best_config]['f1']*100:.1f}%)")
    print()

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved ablation results to results/ablation_results.json")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudarshana Ablation Study")
    parser.add_argument("--weights", "--fusion_ckpt", dest="weights", type=str, default="weights/fusion_best.pt",
                        help="Path to trained fusion model weights")
    parser.add_argument("--clean_dir", type=str, default="data/clean",
                        help="Path to clean dataset")
    parser.add_argument("--stego_dir", type=str, default="data/stego",
                        help="Path to stego dataset")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to test dataset (optional fallback)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clean_path = Path(args.clean_dir)
    stego_path = Path(args.stego_dir)

    # Check if real data exists
    if clean_path.exists() and stego_path.exists() and len(list(clean_path.glob("*.*"))) > 0:
        print(f"Loading real test dataset from {clean_path} & {stego_path}...")
        sys.path.append("src")
        from data_pipeline.dataset import get_dataloaders
        _, _, test_loader = get_dataloaders(
            clean_dir=str(clean_path),
            stego_dir=str(stego_path),
            batch_size=args.batch_size,
            num_workers=2
        )
        run_ablation_study(args.weights, test_loader, device)
    else:
        print(f"\n⚠️  Dataset not found at: {clean_path} and {stego_path}")
        print("  Running a quick smoke test with random data instead...\n")
        from torch.utils.data import DataLoader, TensorDataset
        dummy_images = torch.randn(64, 1, 256, 256)
        dummy_labels = torch.randint(0, 2, (64,))
        dummy_dataset = TensorDataset(dummy_images, dummy_labels)
        dataloader = DataLoader(dummy_dataset, batch_size=args.batch_size)
        run_ablation_study(args.weights, dataloader, device)
