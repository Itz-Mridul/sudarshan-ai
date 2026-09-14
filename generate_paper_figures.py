#!/usr/bin/env python3
"""
generate_paper_figures.py
=========================
Generates ALL figures required for the research paper in one command.

Outputs to results/paper_figures/:
  fig1_roc_comparison.png     — ROC curves for all models (main result)
  fig2_confusion_matrix.png   — Confusion matrix for fusion model
  fig3_training_curves.png    — Loss + accuracy curves for all branches
  fig4_ablation_bar.png       — Bar chart: solo branches vs fusion accuracy
  fig5_quantization_table.png — Table image for size/latency comparison
  paper_results.json          — All numbers in JSON (copy into paper tables)

Run after all models are trained:
  python generate_paper_figures.py
"""

import os, sys, json, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.config import Config
from data_pipeline.dataset import get_dataloaders
from branches.branch_a_pixel import BranchAClassifier
from branches.branch_b_dct import BranchBClassifier
from branches.branch_c_stats import BranchCClassifier
from model.fusion_model import MultiBranchSteganalyzer

OUT_DIR = ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "branch_a": "#E74C3C",
    "branch_b": "#3498DB",
    "branch_c": "#2ECC71",
    "fusion":   "#9B59B6",
    "xunet":    "#F39C12",
    "random":   "#95A5A6",
}


# ─── Model Loader ─────────────────────────────────────────────────────────────

def load_model(checkpoint: str, mode: str, cfg: Config):
    if mode == "branch_a":
        m = BranchAClassifier(feature_dim=cfg.feature_dim_a, num_classes=cfg.num_classes)
    elif mode == "branch_b":
        m = BranchBClassifier(feature_dim=cfg.feature_dim_b, num_classes=cfg.num_classes)
    elif mode == "branch_c":
        m = BranchCClassifier(feature_dim=cfg.feature_dim_c, num_classes=cfg.num_classes)
    else:
        m = MultiBranchSteganalyzer(
            feature_dim_a=cfg.feature_dim_a,
            feature_dim_b=cfg.feature_dim_b,
            feature_dim_c=cfg.feature_dim_c,
        )
    state = torch.load(checkpoint, map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def run_inference(model, loader):
    all_labels, all_preds, all_probs = [], [], []
    for clean_imgs, stego_imgs in loader:
        B = clean_imgs.size(0)
        images = torch.cat([clean_imgs, stego_imgs], dim=0)
        labels = torch.cat([torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)], dim=0)
        
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(logits.argmax(dim=1).numpy())
        all_probs.extend(probs[:, 1].numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    return {
        "accuracy":  accuracy_score(labels, preds) * 100,
        "precision": precision_score(labels, preds, zero_division=0) * 100,
        "recall":    recall_score(labels, preds, zero_division=0) * 100,
        "f1":        f1_score(labels, preds, zero_division=0) * 100,
        "auc":       auc(fpr, tpr) * 100,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "cm":  confusion_matrix(labels, preds).tolist(),
    }


# ─── Figure 1: ROC Curves ─────────────────────────────────────────────────────

def fig1_roc(results: dict):
    fig, ax = plt.subplots(figsize=(7, 6))

    for (name, color_key), metrics in results.items():
        fpr = np.array(metrics["fpr"])
        tpr = np.array(metrics["tpr"])
        ax.plot(fpr, tpr, color=COLORS.get(color_key, "#333"), lw=2,
                label=f"{name} (AUC={metrics['auc']:.1f}%)")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=50%)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Steganalysis Model Comparison\n(BOSSBase v1.01, LSB @ 10% capacity)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "fig1_roc_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig1 saved → {out}")


# ─── Figure 2: Confusion Matrix ───────────────────────────────────────────────

def fig2_confusion(metrics: dict, model_name: str):
    cm = np.array(metrics["cm"])
    acc = metrics["accuracy"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["CLEAN", "STEGO"],
        yticklabels=["CLEAN", "STEGO"],
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 16, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name}\n(Accuracy: {acc:.1f}%)", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "fig2_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig2 saved → {out}")


# ─── Figure 3: Training Curves (all branches) ─────────────────────────────────

def fig3_training_curves():
    history_files = {
        "Branch A (Pixel)": ROOT / "results" / "history_branch_a.json",
        "Branch B (DCT)":   ROOT / "results" / "history_branch_b.json",
        "Branch C (Stats)": ROOT / "results" / "history_branch_c.json",
        "Fusion":           ROOT / "results" / "history_fusion.json",
    }
    color_map = {
        "Branch A (Pixel)": COLORS["branch_a"],
        "Branch B (DCT)":   COLORS["branch_b"],
        "Branch C (Stats)": COLORS["branch_c"],
        "Fusion":           COLORS["fusion"],
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    found_any = False
    for name, path in history_files.items():
        if not path.exists():
            print(f"  ℹ️  No history for {name} yet — skipping")
            continue
        found_any = True
        with open(path) as f:
            h = json.load(f)
        epochs = range(1, len(h["train_loss"]) + 1)
        color = color_map[name]
        ax1.plot(epochs, h["train_loss"], "--", color=color, alpha=0.5, lw=1)
        ax1.plot(epochs, h["val_loss"],   "-",  color=color, lw=2, label=name)
        ax2.plot(epochs, h["train_acc"],  "--", color=color, alpha=0.5, lw=1)
        ax2.plot(epochs, h["val_acc"],    "-",  color=color, lw=2, label=name)

    if not found_any:
        print("⚠️  No training history files found yet — fig3 skipped")
        plt.close()
        return

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Validation Loss (solid) | Train Loss (dashed)")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy (solid) | Train Accuracy (dashed)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — All MBCSS Components", fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "fig3_training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig3 saved → {out}")


# ─── Figure 4: Ablation Bar Chart ─────────────────────────────────────────────

def fig4_ablation_bar(results: dict):
    names  = [n for n, _ in results.keys()]
    accs   = [m["accuracy"] for m in results.values()]
    colors = [COLORS.get(ck, "#888") for _, ck in results.keys()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accs, color=colors, width=0.55, edgecolor="white", linewidth=1.5)

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Ablation Study — Per-Branch vs. Fusion Accuracy\n(BOSSBase v1.01, LSB stego)")
    ax.set_ylim(40, 100)
    ax.axhline(50, color="gray", linestyle="--", lw=1, alpha=0.5, label="Random baseline")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "fig4_ablation_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ fig4 saved → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = Config()

    # Load validation data (reuse val split — same seed as training)
    print("Loading validation data...")
    _, _, test_loader = get_dataloaders(
        clean_dir=cfg.clean_dir,
        stego_dir=cfg.stego_dir,
        batch_size=cfg.batch_size,
        crop_size=cfg.image_size,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        max_images=None,
    )

    checkpoints = {
        ("Branch A (Pixel CNN)", "branch_a"):   ("weights/branch_a_best.pt", "branch_a"),
        ("Branch B (DCT CNN)",   "branch_b"):   ("weights/branch_b_best.pt", "branch_b"),
        ("Branch C (Stats MLP)", "branch_c"):   ("weights/branch_c_best.pt", "branch_c"),
        ("MBCSS Fusion (ours)",  "fusion"):     ("weights/fusion_best.pt",   "fusion"),
    }

    results = {}
    for (name, color_key), (ckpt, mode) in checkpoints.items():
        if not Path(ckpt).exists():
            print(f"  ⏳ {name}: checkpoint not ready yet — skipping ({ckpt})")
            continue
        print(f"  Evaluating {name}...")
        model = load_model(ckpt, mode, cfg)
        labels, preds, probs = run_inference(model, test_loader)
        metrics = compute_metrics(labels, preds, probs)
        results[(name, color_key)] = metrics
        print(f"    Acc={metrics['accuracy']:.1f}%  F1={metrics['f1']:.1f}%  AUC={metrics['auc']:.1f}%")

    if not results:
        print("\n⚠️  No checkpoints found yet. Run training first:")
        print("   python train_all.py")
        print("\nGenerating training curves from any available history files...")
        fig3_training_curves()
        return

    # Save all metrics to JSON (copy numbers into paper tables)
    json_out = ROOT / "results" / "paper_results.json"
    serializable = {f"{n} ({ck})": m for (n, ck), m in results.items()}
    # Remove fpr/tpr from JSON (too large)
    for k in serializable:
        serializable[k] = {mk: mv for mk, mv in serializable[k].items()
                           if mk not in ("fpr", "tpr")}
    with open(json_out, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n📊 All metrics saved → {json_out}")

    # Print comparison table
    print("\n" + "="*75)
    print(f"  {'Model':<28} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7}")
    print("-"*75)
    for (name, _), m in results.items():
        print(f"  {name:<28} {m['accuracy']:>6.1f}% {m['precision']:>6.1f}% "
              f"{m['recall']:>6.1f}% {m['f1']:>6.1f}% {m['auc']:>6.1f}%")
    print("="*75)

    # Generate all figures
    print("\nGenerating paper figures...")
    fig1_roc(results)
    # Confusion matrix for fusion (or best available)
    for (name, ck), m in list(results.items())[::-1]:
        fig2_confusion(m, name)
        break
    fig3_training_curves()
    fig4_ablation_bar(results)

    print(f"\n🎉 All figures saved to {OUT_DIR}/")
    print("   Copy paper_results.json numbers into paper_draft.md tables.")


if __name__ == "__main__":
    main()
