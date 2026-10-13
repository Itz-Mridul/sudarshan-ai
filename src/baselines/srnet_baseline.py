"""
srnet_baseline.py
=================
SRNet Baseline — for comparison with StegShield MBCSS Fusion Model

WHAT THIS IS:
    This is a cleaned-up PyTorch implementation of SRNet, the current
    gold-standard baseline for steganalysis (2019, Boroumand et al.).

    We use this to fill our paper's comparison table:
        | Model          | Accuracy | F1   | Model Size |
        | Xu-Net         |   ~82%   | ~80% |   8.2 MB   |
        | Yedroudj-Net   |   ~87%   | ~85% |  18.5 MB   |
        | SRNet          |   ~90%   | ~89% |  12.1 MB   |
        | StegShield     |  >92%?   | >91%?|   ~X MB    |  ← Our Model

REFERENCE CODE (adapted from):
    GitHub: https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet
    Local copy: external_repos/Pytorch-implementation-of-SRNet/model.py

ORIGINAL PAPER:
    Boroumand, M., Chen, M., & Fridrich, J. (2019).
    "Deep Residual Network for Steganalysis of Digital Images."
    IEEE Transactions on Information Forensics and Security,
    14(5), 1181-1193.
    DOI: 10.1109/TIFS.2018.2871749

KEY DIFFERENCES vs StegShield MBCSS:
    - SRNet: Single-domain (spatial only), 12-layer deep residual network
    - StegShield: Three-domain (spatial + frequency + statistical)
    - SRNet: GPU-only deployment (no INT8 quantization)
    - StegShield: CPU-deployable via INT8 quantization
    - SRNet: Binary output only (clean/stego)
    - StegShield: Binary output + risk score + context scoring

HOW TO RUN:
    # Train SRNet baseline (for paper comparison):
    python src/baselines/srnet_baseline.py --train --epochs 50

    # Evaluate trained SRNet:
    python src/baselines/srnet_baseline.py --eval --checkpoint weights/srnet_best.pt
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
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_pipeline.dataset import get_dataloaders


# ─── SRNet Architecture ───────────────────────────────────────────────────────
# Adapted from: external_repos/Pytorch-implementation-of-SRNet/model.py
# Source: https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet
# Paper:  Boroumand et al., IEEE TIFS 2019

class SRNetResBlock(nn.Module):
    """
    Standard residual block used in SRNet Layers 3–7.
    Both convolutions keep the same spatial size; no downsampling.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual   # skip connection


class SRNetDownBlock(nn.Module):
    """
    Downsampling residual block used in SRNet Layers 8–11.
    Spatial resolution is halved; channels are expanded.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # Shortcut (1x1 conv to match dimensions)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        # Main path
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.pool  = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return out + shortcut


class SRNet(nn.Module):
    """
    Full SRNet model as described in Boroumand et al. 2019.

    Architecture summary:
        Layer 1: Conv(1→64, 3x3) → BN → ReLU                [content suppression]
        Layer 2: Conv(64→16, 3x3) → BN → ReLU               [channel reduction]
        Layers 3–7: 5x ResBlock(16)                           [feature learning]
        Layer 8: DownBlock(16→16) — stride 2 pooling          [spatial reduction]
        Layer 9: DownBlock(16→64)                             [channel expansion]
        Layer 10: DownBlock(64→128)
        Layer 11: DownBlock(128→256)
        Layer 12: Conv(256→512, 3x3, stride=2) → GlobalAvgPool → FC(512→2)

    Input:  (B, 1, 256, 256) — grayscale image
    Output: (B, 2)           — logits [clean, stego]
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Layer 1: Initial content suppression (like high-pass)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Layer 2: Channel bottleneck
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Layers 3–7: Residual blocks (same resolution)
        self.res_layers = nn.Sequential(
            SRNetResBlock(16),   # Layer 3
            SRNetResBlock(16),   # Layer 4
            SRNetResBlock(16),   # Layer 5
            SRNetResBlock(16),   # Layer 6
            SRNetResBlock(16),   # Layer 7
        )

        # Layers 8–11: Downsampling blocks (resolution halved each time)
        self.down8  = SRNetDownBlock(16,  16)   # Layer 8
        self.down9  = SRNetDownBlock(16,  64)   # Layer 9
        self.down10 = SRNetDownBlock(64,  128)  # Layer 10
        self.down11 = SRNetDownBlock(128, 256)  # Layer 11

        # Layer 12: Final conv + global average pool
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.res_layers(x)
        x = self.down8(x)
        x = self.down9(x)
        x = self.down10(x)
        x = self.down11(x)
        x = self.layer12(x)
        x = x.mean(dim=(2, 3))   # Global average pooling
        return self.classifier(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Training & Evaluation ───────────────────────────────────────────────────

def train_srnet(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SRNet Baseline Training")
    print(f"  Reference: github.com/brijeshiitg/Pytorch-implementation-of-SRNet")
    print(f"  Paper: Boroumand et al., IEEE TIFS 2019")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    model = SRNet().to(device)
    print(f"SRNet parameters: {model.count_params():,}")

    train_loader, val_loader, _ = get_dataloaders(
        clean_dir=args.clean_dir,
        stego_dir=args.stego_dir,
        batch_size=32,
        crop_size=256,
        max_images=args.max_images
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    os.makedirs("weights", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for clean_imgs, stego_imgs in train_loader:
            B = clean_imgs.size(0)
            images = torch.cat([clean_imgs, stego_imgs], dim=0).to(device)
            labels = torch.cat([torch.zeros(B, dtype=torch.long),
                                torch.ones(B, dtype=torch.long)], dim=0).to(device)
            idx = torch.randperm(2 * B)
            images, labels = images[idx], labels[idx]

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss    += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total   += images.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for clean_imgs, stego_imgs in val_loader:
                B = clean_imgs.size(0)
                images = torch.cat([clean_imgs, stego_imgs], dim=0).to(device)
                labels = torch.cat([torch.zeros(B, dtype=torch.long),
                                    torch.ones(B, dtype=torch.long)], dim=0).to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += images.size(0)

        tr_acc = train_correct / train_total * 100
        vl_acc = val_correct   / val_total   * 100
        vl_ls  = val_loss      / val_total

        print(f"Epoch [{epoch:3d}/{args.epochs}]  "
              f"Train acc: {tr_acc:.1f}%  "
              f"Val acc: {vl_acc:.1f}%  "
              f"Val loss: {vl_ls:.4f}  "
              f"({time.time()-t0:.0f}s)")

        if vl_ls < best_val_loss:
            best_val_loss = vl_ls
            torch.save(model.state_dict(), "weights/srnet_best.pt")
            print(f"  ★ New best SRNet saved  (val_loss={best_val_loss:.4f})")

    size_mb = Path("weights/srnet_best.pt").stat().st_size / 1e6
    print(f"\n✅ SRNet training complete!")
    print(f"   Best val accuracy: Check weights/srnet_best.pt")
    print(f"   Model size: {size_mb:.1f} MB")
    print(f"\n   → Run StegShield comparison:")
    print(f"     python src/training/evaluate.py --checkpoint weights/srnet_best.pt --mode srnet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SRNet Baseline Training")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--max_images",  type=int, default=None)
    parser.add_argument("--clean_dir",   type=str, default="data/clean")
    parser.add_argument("--stego_dir",   type=str, default="data/stego")
    parser.add_argument("--checkpoint",  type=str, default="weights/srnet_best.pt")
    args = parser.parse_args()

    if args.train:
        train_srnet(args)
    elif args.eval:
        print("Eval mode: coming in next version. Run evaluate.py instead.")
    else:
        parser.print_help()
