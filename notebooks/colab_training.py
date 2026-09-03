"""
colab_training.py
=================
Google Colab Training Script — MBCSS Multi-Branch Steganalysis System

USAGE ON GOOGLE COLAB:
  1. Upload your entire EDI_Project folder to Google Drive
  2. Open Google Colab (colab.research.google.com)
  3. Select Runtime → Change Runtime Type → T4 GPU
  4. In Colab, run this cell first:

     from google.colab import drive
     drive.mount('/content/drive')

     import subprocess, os
     project_path = '/content/drive/MyDrive/EDI_Project'
     os.chdir(project_path)
     !pip install -r requirements.txt -q

  5. Then run this script:
     !python notebooks/colab_training.py --mode all

WHAT IT DOES:
  Trains all 3 branches + fusion in the correct order.
  Saves checkpoints to weights/ every 5 epochs.
  Plots loss + accuracy curves to results/.

TRAINING ORDER (as per PRD RULE-ARCH-03):
  Phase 1: Train Branch A (pixel) alone      → weights/branch_a_best.pt
  Phase 2: Train Branch B (DCT) alone        → weights/branch_b_best.pt
  Phase 3: Train Branch C (stats) alone      → weights/branch_c_best.pt
  Phase 4: Load branches, train fusion head  → weights/fusion_best.pt
  Phase 5: Fine-tune everything jointly (low LR) → weights/fusion_finetuned.pt

EXPECTED TIMES (Colab T4 GPU, 10k images):
  Branch A: ~1.5-2 hours
  Branch B: ~2-3 hours (DCT computation is expensive)
  Branch C: ~30 mins (MLP, fast)
  Fusion:   ~1 hour
  Fine-tune:~30 mins
  Total:    ~6-8 hours (can be split across sessions)
"""

import os
import sys
import argparse
import subprocess

# Change to project root (works when called from any directory in Colab)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

import torch
import json


def check_gpu():
    """Print GPU info for Colab."""
    print("\n" + "=" * 55)
    print("  ENVIRONMENT CHECK")
    print("=" * 55)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {mem:.1f} GB")
        print("  ✅ GPU available — training will be fast!")
    else:
        print("  ⚠️  No GPU detected — training on CPU (slower)")
        print("  Tip: Runtime → Change Runtime Type → T4 GPU")

    print(f"\n  Working directory: {os.getcwd()}")
    clean_count = len(list(os.scandir("data/clean"))) if os.path.exists("data/clean") else 0
    stego_count = len(list(os.scandir("data/stego"))) if os.path.exists("data/stego") else 0
    print(f"  Clean images: {clean_count:,}")
    print(f"  Stego images: {stego_count:,}")

    if clean_count == 0 or stego_count == 0:
        print("\n  ⚠️  DATASET NOT READY! Run setup first:")
        print("     bash scripts/setup_boss_dataset.sh")
    else:
        print("  ✅ Dataset ready for training!")
    print("=" * 55 + "\n")


def run_phase(phase: str, extra_args: list = None):
    """
    Launch one training phase using src/training/train.py.
    Streams output to console.
    """
    cmd = [sys.executable, "src/training/train.py", "--mode", phase]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*55}")
    print(f"  STARTING: {phase.upper()} training")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*55}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"\n✅ Phase '{phase}' completed successfully!")
    else:
        print(f"\n❌ Phase '{phase}' failed (exit code {result.returncode})")
        print("   Check the output above for the error.")
    return result.returncode == 0


def quick_test(max_images: int = 200, epochs: int = 2):
    """
    Quick 2-epoch test on 200 images to verify the pipeline works.
    Should complete in ~5 minutes on CPU, ~1 minute on GPU.
    """
    print("\n" + "=" * 55)
    print("  QUICK PIPELINE TEST (200 images, 2 epochs)")
    print("  If this passes, full training will also work.")
    print("=" * 55)

    for mode in ["branch_a", "branch_b", "branch_c", "fusion"]:
        success = run_phase(mode, [
            "--max_images", str(max_images),
            "--epochs", str(epochs)
        ])
        if not success:
            print(f"\n❌ Quick test FAILED on {mode}. Fix errors before full training.")
            return False

    print("\n✅ Quick test PASSED! All 4 modes work.")
    print("  Ready for full training (remove --max_images).")
    return True


def train_all(epochs_per_branch: int = 50, epochs_fusion: int = 40, epochs_finetune: int = 20):
    """
    Full training pipeline — all phases in order.
    """
    check_gpu()

    print("\n" + "=" * 55)
    print("  FULL TRAINING PIPELINE")
    print(f"  Branch epochs:  {epochs_per_branch}")
    print(f"  Fusion epochs:  {epochs_fusion}")
    print(f"  Finetune epochs:{epochs_finetune}")
    print("=" * 55)

    # Phase 1-3: Train each branch independently
    for mode, target_acc in [("branch_a", "82%"), ("branch_b", "76%"), ("branch_c", "68%")]:
        print(f"\n{'─'*55}")
        print(f"  Phase: {mode} (target: {target_acc}+ accuracy)")
        run_phase(mode, ["--epochs", str(epochs_per_branch)])

    # Phase 4: Train fusion (loads pretrained branch weights)
    print(f"\n{'─'*55}")
    print("  Phase: fusion (loads pretrained branches)")
    run_phase("fusion", ["--epochs", str(epochs_fusion)])

    # Phase 5: Fine-tune everything jointly at very low LR
    print(f"\n{'─'*55}")
    print("  Phase: joint fine-tuning (all branches + fusion, lr=1e-5)")
    run_phase("joint", [
        "--epochs", str(epochs_finetune),
        "--lr", "0.00001"
    ])

    print("\n" + "=" * 55)
    print("  🎉 FULL TRAINING COMPLETE!")
    print("  Saved to: weights/")
    print("  Next: python src/training/evaluate.py --ablation")
    print("=" * 55)


def evaluate_all():
    """Run full ablation study evaluation."""
    print("\nRunning ablation study evaluation...")
    subprocess.run([
        sys.executable, "src/training/evaluate.py", "--ablation"
    ])


def generate_colab_notebook_code():
    """
    Print the Colab cell code to paste into a notebook.
    Copy-paste this into Google Colab cells.
    """
    notebook_code = '''
# ═══════════════════════════════════════════════════════════════
# CELL 1: Mount Google Drive and navigate to project
# ═══════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os, sys
PROJECT = '/content/drive/MyDrive/EDI_Project'  # ← CHANGE THIS PATH IF NEEDED
os.chdir(PROJECT)
print("Working in:", os.getcwd())


# ═══════════════════════════════════════════════════════════════
# CELL 2: Install dependencies
# ═══════════════════════════════════════════════════════════════

!pip install -r requirements.txt -q
import torch
print(f"PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Setup dataset (BOSS + LSB stego pairs)
# ═══════════════════════════════════════════════════════════════

!bash scripts/setup_boss_dataset.sh


# ═══════════════════════════════════════════════════════════════
# CELL 4: Quick test (200 images, 2 epochs — verify pipeline)
# ═══════════════════════════════════════════════════════════════

!python notebooks/colab_training.py --mode quick_test


# ═══════════════════════════════════════════════════════════════
# CELL 5a: Train Branch A (pixel CNN + SRM) — ~2 hours on T4
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode branch_a --epochs 50


# ═══════════════════════════════════════════════════════════════
# CELL 5b: Train Branch B (DCT CNN) — ~3 hours on T4
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode branch_b --epochs 50


# ═══════════════════════════════════════════════════════════════
# CELL 5c: Train Branch C (stats MLP) — ~30 mins on T4
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode branch_c --epochs 50


# ═══════════════════════════════════════════════════════════════
# CELL 6: Train Fusion (loads pretrained branches) — ~1 hour
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode fusion --epochs 40


# ═══════════════════════════════════════════════════════════════
# CELL 7: Evaluate + Ablation Study (generates paper tables)
# ═══════════════════════════════════════════════════════════════

!python src/training/evaluate.py --ablation


# ═══════════════════════════════════════════════════════════════
# CELL 8: Train Xu-Net baseline (for comparison table)
# ═══════════════════════════════════════════════════════════════

!python src/baselines/xunet_baseline.py --train --epochs 50


# ═══════════════════════════════════════════════════════════════
# CELL 9: Quantize for CPU deployment
# ═══════════════════════════════════════════════════════════════

!python src/deploy/quantize.py --model_path weights/fusion_best.pt


# ═══════════════════════════════════════════════════════════════
# CELL 10: Download weights to local machine
# ═══════════════════════════════════════════════════════════════

import shutil
shutil.make_archive('/content/drive/MyDrive/EDI_weights_backup', 'zip', 'weights/')
print("✅ Weights backed up to Google Drive!")
'''
    print(notebook_code)
    # Also save to a file
    with open("notebooks/colab_cells.txt", "w") as f:
        f.write("# Copy-paste these cells into Google Colab\n")
        f.write(notebook_code)
    print("\n✅ Saved to: notebooks/colab_cells.txt")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Colab training coordinator for MBCSS"
    )
    parser.add_argument("--mode", default="check",
                        choices=["check", "quick_test", "branch_a", "branch_b",
                                 "branch_c", "fusion", "joint", "all",
                                 "evaluate", "print_colab_code"],
                        help="What to run")
    parser.add_argument("--epochs_branch", type=int, default=50)
    parser.add_argument("--epochs_fusion", type=int, default=40)
    parser.add_argument("--epochs_finetune", type=int, default=20)
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "check":
        check_gpu()
    elif args.mode == "quick_test":
        quick_test(max_images=200, epochs=2)
    elif args.mode == "all":
        train_all(args.epochs_branch, args.epochs_fusion, args.epochs_finetune)
    elif args.mode == "evaluate":
        evaluate_all()
    elif args.mode == "print_colab_code":
        generate_colab_notebook_code()
    elif args.mode in ("branch_a", "branch_b", "branch_c", "fusion", "joint"):
        extra = []
        if args.max_images:
            extra += ["--max_images", str(args.max_images)]
        run_phase(args.mode, extra)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
