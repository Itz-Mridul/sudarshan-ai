"""
visualize_residuals.py
======================
Step 4 of the Checklist — "Look at the data with your own eyes"

WHAT THIS DOES:
  Computes and plots pixel residual maps side-by-side for clean vs stego images.
  This becomes Figure 1 in your paper.

HOW TO RUN:
  python notebooks/visualize_residuals.py \
    --clean data/clean \
    --stego data/stego \
    --n 4

  Saves output to: results/residual_visualization.png
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without a display)
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ─── Residual Computation ─────────────────────────────────────────────────────

def compute_residual(image_array: np.ndarray, scale: int = 10) -> np.ndarray:
    """
    Compute pixel residual map using a simple high-pass filter.
    Simulates what Branch A's SRM layer amplifies.

    Method: subtract a 3×3 average neighbourhood from the pixel value.
    This leaves only high-frequency noise — exactly where stego hides.

    Args:
        image_array: grayscale image as numpy (H, W) uint8
        scale      : amplification factor for visibility (default 10×)

    Returns:
        residual map as (H, W) float32, range roughly [-1, 1]
    """
    img = image_array.astype(np.float32) / 255.0

    # Simple 3x3 average filter (approximation of what SRM does)
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(img, size=3)
    residual = img - smoothed           # high-pass residual

    # Amplify and clip for display
    residual_amp = np.clip(residual * scale, -1.0, 1.0)
    return residual_amp


def load_image_as_array(path: str, size: int = 256) -> np.ndarray:
    """Load image as grayscale uint8 numpy array."""
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_residual_comparison(
    clean_dir: str,
    stego_dir: str,
    n_images: int = 4,
    output_path: str = "results/residual_visualization.png"
) -> None:
    """
    Plot a grid comparing:
      Row 1: Clean images
      Row 2: Clean pixel residuals
      Row 3: Stego images
      Row 4: Stego pixel residuals

    This visually proves that stego introduces high-frequency noise
    that is invisible to human eyes but visible in the residual map.

    Args:
        clean_dir   : folder of clean images (data/clean)
        stego_dir   : folder of stego images (data/stego)
        n_images    : number of image pairs to show
        output_path : where to save the figure
    """
    clean_path = Path(clean_dir)
    stego_path = Path(stego_dir)

    exts = {".png", ".pgm", ".jpg", ".jpeg"}
    clean_files = sorted([f for f in clean_path.iterdir() if f.suffix.lower() in exts])[:n_images]
    stego_files = sorted([f for f in stego_path.iterdir() if f.suffix.lower() in exts])[:n_images]

    if not clean_files:
        print(f"❌ No images found in {clean_dir}")
        return
    if not stego_files:
        print(f"❌ No images found in {stego_dir}")
        return

    n = min(len(clean_files), len(stego_files), n_images)
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 16))
    fig.suptitle(
        "Pixel Residual Maps: Clean vs Stego\n"
        "Stego residuals show structured high-frequency noise (invisible to eyes, visible here)",
        fontsize=14, fontweight="bold", y=0.98
    )

    row_labels = [
        "Clean Image",
        "Clean Residual\n(should look like white noise)",
        "Stego Image\n(looks identical to clean)",
        "Stego Residual\n(structured noise = hidden data)"
    ]

    for col in range(n):
        # Load images
        clean_arr = load_image_as_array(str(clean_files[col]))
        stego_arr = load_image_as_array(str(stego_files[col]))

        # Compute residuals
        clean_res = compute_residual(clean_arr)
        stego_res = compute_residual(stego_arr)

        # Row 0: Clean image
        axes[0, col].imshow(clean_arr, cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(clean_files[col].stem, fontsize=8)
        axes[0, col].axis("off")

        # Row 1: Clean residual
        im1 = axes[1, col].imshow(clean_res, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[1, col].axis("off")

        # Row 2: Stego image (should look IDENTICAL to clean)
        axes[2, col].imshow(stego_arr, cmap="gray", vmin=0, vmax=255)
        axes[2, col].set_title(stego_files[col].stem, fontsize=8)
        axes[2, col].axis("off")

        # Row 3: Stego residual (should show structure)
        im3 = axes[3, col].imshow(stego_res, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
        axes[3, col].axis("off")

    # Add row labels on left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].set_ylabel(label, fontsize=10, rotation=0,
                                     labelpad=120, va="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✅ Residual visualization saved to: {output_path}")
    print(f"   This is your Figure 1 for the paper.")
    plt.close()


def check_pixel_differences(clean_path: str, stego_path: str) -> None:
    """
    Print a simple stat report confirming steganography modified the LSBs.
    Use this to VERIFY your LSB embedder worked correctly.
    """
    clean = load_image_as_array(clean_path)
    stego = load_image_as_array(stego_path)

    diff = np.abs(clean.astype(int) - stego.astype(int))
    n_changed = np.sum(diff > 0)
    max_diff = np.max(diff)
    pct_changed = n_changed / clean.size * 100

    print("\nLSB Embedding Verification Report:")
    print(f"  Total pixels       : {clean.size:,}")
    print(f"  Pixels changed     : {n_changed:,}  ({pct_changed:.1f}%)")
    print(f"  Max pixel diff     : {max_diff}  (should always be 0 or 1 for LSB)")
    print(f"  Human-visible?     : {'NO — max diff is 1' if max_diff <= 1 else 'WARNING — diff > 1!'}")
    print(f"  Embedding confirmed: {'✅ YES' if n_changed > 0 and max_diff <= 1 else '❌ CHECK YOUR EMBEDDER'}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize pixel residual maps for clean vs stego images"
    )
    parser.add_argument("--clean",  default="data/clean",
                        help="Folder of clean images")
    parser.add_argument("--stego",  default="data/stego",
                        help="Folder of stego images")
    parser.add_argument("--n",      type=int, default=4,
                        help="Number of image pairs to show")
    parser.add_argument("--output", default="results/residual_visualization.png",
                        help="Output figure path")
    parser.add_argument("--verify", action="store_true",
                        help="Also verify LSB embedding on first pair")
    args = parser.parse_args()

    plot_residual_comparison(args.clean, args.stego, args.n, args.output)

    if args.verify:
        clean_files = sorted(Path(args.clean).glob("*.png"))
        stego_files = sorted(Path(args.stego).glob("*.png"))
        if clean_files and stego_files:
            check_pixel_differences(str(clean_files[0]), str(stego_files[0]))


if __name__ == "__main__":
    main()
