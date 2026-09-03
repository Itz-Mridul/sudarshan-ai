"""
xunet_baseline.py
=================
Xu-Net Baseline — PyTorch Implementation

PURPOSE:
  This is a SIMPLIFIED PyTorch reimplementation of:
  G. Xu, H.-Z. Wu, and Y.-Q. Shi,
  "Structural Design of Convolutional Neural Networks for Steganalysis,"
  IEEE Signal Processing Letters, vol. 23, no. 5, pp. 708–712, 2016.

  Used to generate the COMPARISON TABLE in the paper (Section 5).
  You must train this separately and compare results against MBCSS.

  The original Xu-Net uses TensorFlow 1.x (see external_repos/).
  This reimplementation uses the SAME architecture but in PyTorch
  so it can be trained and evaluated under identical conditions.

KEY DESIGN CHOICES (exactly from the paper):
  1. First layer: SRM high-pass filter (30 kernels, 5×5, FROZEN)
  2. ABS layer:   absolute value activation after SRM (not ReLU)
  3. TanH:        hyperbolic tangent in first two conv blocks
  4. 1×1 conv:    in deeper layers for channel mixing without spatial info loss
  5. Final: Global average pool → FC(256) → FC(2) (clean/stego)

HOW TO TRAIN:
  python src/baselines/xunet_baseline.py --train \
    --clean_dir data/clean --stego_dir data/stego

HOW TO EVALUATE (for paper Table):
  python src/baselines/xunet_baseline.py --evaluate \
    --checkpoint weights/xunet_best.pt \
    --clean_dir data/clean --stego_dir data/stego
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.srm_filters import SRMFilterLayer
from data_pipeline.dataset import get_dataloaders
from training.config import Config


# ─── Xu-Net Architecture ──────────────────────────────────────────────────────

class XuNet(nn.Module):
    """
    Xu-Net: CNN for Steganalysis (Xu et al., 2016).

    Architecture follows Figure 2 from the paper:
      SRM(30) → ABS → TanH Block 1 → TanH Block 2 → ReLU Block 3 →
      1×1 Conv Block 4 → Global AvgPool → FC(256) → FC(2)

    IMPORTANT: This is YOUR BASELINE model to beat.
    Train this on the same dataset, with the same split, and compare.
    If MBCSS doesn't beat Xu-Net by ≥2%, revise your approach.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # ── SRM Pre-Processing Layer (fixed, from Fridrich 2012) ──────────────
        # Key Xu-Net innovation: use existing physics-derived SRM filters
        # instead of learning convolutions from scratch
        self.srm = SRMFilterLayer()   # 30 frozen filters → (batch, 30, H, W)

        # ── Block 1: ABS + TanH activation ───────────────────────────────────
        # The ABS activation (|x|) after SRM is the primary Xu-Net innovation.
        # It makes residuals symmetric around 0, improving statistical modeling.
        self.conv1 = nn.Conv2d(30, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        # Activation: TanH with saturation clamp (controls overfitting in early layers)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ── Block 2: TanH activation ──────────────────────────────────────────
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ── Block 3: ReLU activation ──────────────────────────────────────────
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        # ── Block 4: 1×1 convolutions (reduce channels, not spatial) ─────────
        # 1×1 conv mixes channels without using spatial information.
        # This "reduces the strength of modeling" in deeper layers (per paper).
        self.conv4_1x1 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.bn4       = nn.BatchNorm2d(256)

        # ── Global Average Pooling ────────────────────────────────────────────
        # Makes model input-size agnostic
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier ────────────────────────────────────────────────────────
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: grayscale image tensor (batch, 1, H, W), values in [-1, 1]

        Returns:
            logits (batch, 2)
        """
        # Step 1: SRM residuals
        x = self.srm(x)            # (batch, 30, H, W) — pixel noise residuals

        # Step 2: ABS activation (the Xu-Net key innovation!)
        x = torch.abs(x)           # all residuals become positive

        # Step 3: Block 1 — TanH activation (clamps range, reduces overfitting)
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = self.pool1(x)          # H/2, W/2

        # Step 4: Block 2 — TanH
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.pool2(x)          # H/4, W/4

        # Step 5: Block 3 — ReLU (standard from here)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)          # H/8, W/8

        # Step 6: Block 4 — 1×1 convolution
        x = F.relu(self.bn4(self.conv4_1x1(x)))

        # Step 7: Global Average Pool + Classify
        x = self.gap(x)            # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 256)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)            # (batch, 2)

        return x


# ─── Training ─────────────────────────────────────────────────────────────────

def train_xunet(
    clean_dir: str = "data/clean",
    stego_dir: str = "data/stego",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_images: int = None,
    checkpoint_dir: str = "weights"
) -> dict:
    """
    Train the Xu-Net baseline and save the best checkpoint.

    Returns training history dict (for plotting in comparison table).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Xu-Net Baseline on: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        clean_dir=clean_dir,
        stego_dir=stego_dir,
        batch_size=batch_size,
        crop_size=256,
        val_split=0.2,
        max_images=max_images
    )

    # Model
    model = XuNet(num_classes=2).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Xu-Net: {total_p:,} total params | {train_p:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "xunet_best.pt")

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item() * images.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += images.size(0)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                v_loss += loss.item() * images.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total += images.size(0)

        train_acc = t_correct / t_total * 100
        val_acc   = v_correct / v_total * 100
        val_loss  = v_loss / v_total

        history["train_loss"].append(t_loss / t_total)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ★ Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                  f"[SAVED] → {ckpt_path}")
        elif epoch % 5 == 0:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

    print(f"\n✅ Xu-Net training complete. Best val loss: {best_val_loss:.4f}")
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_xunet(
    checkpoint_path: str,
    clean_dir: str = "data/clean",
    stego_dir: str = "data/stego",
    max_images: int = None
) -> dict:
    """
    Evaluate a trained Xu-Net and return metrics dict.
    Results go into your paper's comparison table.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    device = torch.device("cpu")
    model = XuNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    _, val_loader = get_dataloaders(
        clean_dir=clean_dir, stego_dir=stego_dir,
        batch_size=32, crop_size=256, val_split=0.2, max_images=max_images
    )

    all_labels, all_preds, all_probs = [], [], []
    for images, labels in val_loader:
        logits = model(images.to(device))
        probs  = F.softmax(logits, dim=1)
        all_labels.extend(labels.numpy())
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall":    recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1":        f1_score(y_true, y_pred, zero_division=0) * 100,
        "auc":       roc_auc_score(y_true, y_prob) * 100,
    }

    print("\n" + "=" * 50)
    print("  Xu-Net Baseline Results (for your comparison table)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k.capitalize():12} : {v:.2f}%")
    print("=" * 50)
    print("  Compare these against MBCSS fusion model results.")
    return metrics


# ─── Self-Test ────────────────────────────────────────────────────────────────

def test_architecture():
    """Verify Xu-Net instantiates and runs a forward pass correctly."""
    print("Testing Xu-Net architecture...")
    model = XuNet(num_classes=2)
    model.eval()

    batch = torch.randn(4, 1, 256, 256)   # 4 grayscale images
    with torch.no_grad():
        logits = model(batch)

    print(f"  Input shape:   {batch.shape}")
    print(f"  Output shape:  {logits.shape}  ← (4, 2)")

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:  {total_p:,}")
    print(f"  Trainable:     {train_p:,}  (SRM frozen)")
    print("  ✅ Xu-Net OK!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xu-Net Baseline (Xu et al., 2016)")
    parser.add_argument("--test",       action="store_true",
                        help="Test architecture only (no data needed)")
    parser.add_argument("--train",      action="store_true",
                        help="Train Xu-Net baseline")
    parser.add_argument("--evaluate",   action="store_true",
                        help="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", type=str, default="weights/xunet_best.pt")
    parser.add_argument("--clean_dir",  type=str, default="data/clean")
    parser.add_argument("--stego_dir",  type=str, default="data/stego")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()

    if args.test:
        test_architecture()
    elif args.train:
        train_xunet(
            clean_dir=args.clean_dir,
            stego_dir=args.stego_dir,
            epochs=args.epochs,
            lr=args.lr,
            max_images=args.max_images
        )
    elif args.evaluate:
        evaluate_xunet(
            checkpoint_path=args.checkpoint,
            clean_dir=args.clean_dir,
            stego_dir=args.stego_dir,
            max_images=args.max_images
        )
    else:
        test_architecture()
