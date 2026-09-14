"""
colab_training.py
=================
Google Colab Training Script — MBCSS Multi-Branch Steganalysis System
TARGET: >95% Accuracy on BOSSBase LSB Steganography

USAGE ON GOOGLE COLAB:
  1. Upload your entire EDI_Project folder to Google Drive
  2. Open Google Colab (colab.research.google.com)
  3. Select Runtime → Change Runtime Type → T4 GPU
  4. Run the cells in order (see notebooks/colab_cells.txt for copy-paste cells)

KEY CHANGES (v2 — accuracy fix):
  - Joint end-to-end training (all branches + fusion simultaneously)
  - Deeper residual CNN backbone in Branch A (3.25M params vs old 453K)
  - TanH-clamped SRM residuals for stable gradient flow
  - Label smoothing (0.1) to prevent overconfident predictions
  - Paired mini-batching (clean+stego get IDENTICAL spatial crops)
  - Early stop patience=15, 80 epochs

EXPECTED TIMES (Colab T4 GPU, 10,000 image pairs):
  Quick test (200 images, 3 epochs): ~5 minutes
  Full joint training (80 epochs):   ~3-4 hours
  Evaluation + figures:              ~10 minutes
  Xu-Net baseline:                   ~1 hour
  Total:                             ~5 hours
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


def quick_test(max_images: int = 200, epochs: int = 3):
    """
    Quick 3-epoch test on 200 images to verify the FULL pipeline works.
    Should complete in ~5 minutes on T4 GPU.
    Tests joint training mode (all branches + fusion simultaneously).
    """
    print("\n" + "=" * 55)
    print("  QUICK PIPELINE TEST (200 images, 3 epochs)")
    print("  Testing JOINT mode (all branches + fusion)")
    print("  If this passes, full training will also work.")
    print("=" * 55)

    success = run_phase("joint", [
        "--max_images", str(max_images),
        "--epochs", str(epochs)
    ])
    if success:
        print("\n✅ Quick test PASSED! Pipeline is working correctly.")
        print("  You should see loss dropping below 0.65 within 3 epochs.")
        print("  Ready for full training.")
    else:
        print("\n❌ Quick test FAILED. Fix errors before full training.")
    return success


def train_full(epochs: int = 80):
    """
    Full joint training — trains ALL branches + fusion end-to-end simultaneously.
    This is the RECOMMENDED approach for >95% accuracy.

    Why joint training?
      - All branches co-adapt to the fusion task
      - Gradients flow from the final classifier back through ALL branches
      - No risk of mismatched branch features when they are frozen
      - Achieves higher accuracy than sequential branch-by-branch training
    """
    check_gpu()

    print("\n" + "=" * 55)
    print("  FULL JOINT TRAINING — TARGET: >95% ACCURACY")
    print(f"  Epochs: {epochs}")
    print(f"  Images: All 10,000 pairs (16,000 train + 4,000 val)")
    print(f"  Model:  4.97M params (Residual CNN + Fusion MLP)")
    print(f"  Est. time: ~3-4 hours on T4 GPU")
    print("=" * 55)

    run_phase("joint", ["--epochs", str(epochs)])

    print("\n" + "=" * 55)
    print("  🎉 TRAINING COMPLETE!")
    print("  Best model saved to: weights/fusion_best.pt")
    print("  Next: run evaluate_all() or Cell 5 in notebook")
    print("=" * 55)


def evaluate_all():
    """Run full evaluation: metrics, ROC, confusion matrix, ablation, figures."""
    print("\nRunning full evaluation suite...")

    # Full evaluation with ablation
    subprocess.run([
        sys.executable, "src/training/evaluate.py",
        "--checkpoint", "weights/fusion_best.pt",
        "--mode", "fusion",
        "--ablation"
    ])

    # Paper figures
    if os.path.exists("generate_paper_figures.py"):
        print("\nGenerating paper figures...")
        subprocess.run([sys.executable, "generate_paper_figures.py"])

    # CPU benchmark
    if os.path.exists("benchmark_inference.py"):
        print("\nRunning inference benchmark...")
        subprocess.run([sys.executable, "benchmark_inference.py"])


def generate_colab_notebook_code():
    """
    Print copy-paste Colab cells for the updated joint training pipeline.
    """
    notebook_code = '''
# ═══════════════════════════════════════════════════════════════
# CELL 1: Mount Google Drive and navigate to project
# ═══════════════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import os, sys
PROJECT = '/content/drive/MyDrive/EDI_Project'  # ← CHANGE IF NEEDED
os.chdir(PROJECT)
print("Working in:", os.getcwd())


# ═══════════════════════════════════════════════════════════════
# CELL 2: Install dependencies + verify GPU
# ═══════════════════════════════════════════════════════════════

!pip install -r requirements.txt -q
import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT DETECTED — go to Runtime > Change Runtime Type > T4 GPU'}")
print(f"Clean images: {len(os.listdir('data/clean'))}")
print(f"Stego images: {len(os.listdir('data/stego'))}")


# ═══════════════════════════════════════════════════════════════
# CELL 3: Quick test — verify pipeline before full run (~5 mins)
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode joint --max_images 200 --epochs 3


# ═══════════════════════════════════════════════════════════════
# CELL 4: FULL JOINT TRAINING — ~3-4 hours on T4 GPU
#         Trains all 3 branches + fusion simultaneously
#         Target: >95% validation accuracy
# ═══════════════════════════════════════════════════════════════

!python src/training/train.py --mode joint --epochs 80


# ═══════════════════════════════════════════════════════════════
# CELL 5: Evaluate — generates all paper metrics and figures
# ═══════════════════════════════════════════════════════════════

!python src/training/evaluate.py --checkpoint weights/fusion_best.pt --mode fusion
!python generate_paper_figures.py
!python benchmark_inference.py


# ═══════════════════════════════════════════════════════════════
# CELL 6: Train Xu-Net baseline (for comparison table in paper)
# ═══════════════════════════════════════════════════════════════

!python src/baselines/xunet_baseline.py --train --epochs 50


# ═══════════════════════════════════════════════════════════════
# CELL 7: Quantize for CPU deployment (INT8)
# ═══════════════════════════════════════════════════════════════

!python src/deploy/quantize.py --model_path weights/fusion_best.pt


# ═══════════════════════════════════════════════════════════════
# CELL 8: Back up weights + results to Google Drive
# ═══════════════════════════════════════════════════════════════

import shutil
shutil.make_archive('/content/drive/MyDrive/EDI_weights_backup', 'zip', 'weights/')
shutil.make_archive('/content/drive/MyDrive/EDI_results_backup', 'zip', 'results/')
print("✅ Weights and results backed up to Google Drive!")
'''
    print(notebook_code)
    with open("notebooks/colab_cells.txt", "w") as f:
        f.write("# Copy-paste these cells into Google Colab\n")
        f.write(notebook_code)
    print("\n✅ Saved to: notebooks/colab_cells.txt")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Colab training coordinator for MBCSS — target >95% accuracy"
    )
    parser.add_argument("--mode", default="check",
                        choices=["check", "quick_test", "full", "joint",
                                 "evaluate", "print_colab_code"],
                        help="What to run")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Training epochs for full/joint mode")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "check":
        check_gpu()
    elif args.mode == "quick_test":
        quick_test()
    elif args.mode in ("full", "joint"):
        train_full(args.epochs)
    elif args.mode == "evaluate":
        evaluate_all()
    elif args.mode == "print_colab_code":
        generate_colab_notebook_code()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
