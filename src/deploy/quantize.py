"""
quantize.py
===========
INT8 Post-Training Quantization — CPU Deployment (Step 12 / Checklist)

WHAT THIS DOES:
  Takes a trained MultiBranchSteganalyzer model and compresses it from
  FP32 to INT8, reducing size ~4× and inference time ~2-3×.
  The quantized model runs on CPU only — perfect for air-gapped DRDO endpoints.

HOW TO USE:
  python src/deploy/quantize.py \
    --model_path weights/fusion_best.pt \
    --output_path weights/fusion_int8.pt \
    --data_dir data/clean \
    --n_calib 200

PRD TARGETS:
  - Model size: ~50MB → ~12MB  (4× reduction)
  - CPU inference: ~200ms → ~60ms per image
  - Accuracy drop: expect 1–5% — report the ACTUAL number, don't assume <2%
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ─── Calibration Dataset ──────────────────────────────────────────────────────

class CalibrationDataset(Dataset):
    """
    Loads a small set of images for calibration.
    During calibration, we feed real data so the quantizer can observe
    the range of activations and choose optimal INT8 scale factors.
    """

    def __init__(self, image_dir: str, n_images: int = 200, image_size: int = 256):
        self.image_size = image_size
        exts = {".png", ".jpg", ".jpeg", ".pgm", ".bmp"}
        paths = sorted(Path(image_dir).glob("*"))
        paths = [p for p in paths if p.suffix.lower() in exts][:n_images]
        if not paths:
            raise FileNotFoundError(f"No images found in: {image_dir}")
        self.paths = paths
        print(f"[Calibration] Using {len(self.paths)} images from {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0)   # (1, H, W)


# ─── Quantization ─────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    """Load the trained fusion model from a .pt checkpoint."""
    # Late import to avoid circular dependency
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from model.fusion_model import MultiBranchSteganalyzer

    model = MultiBranchSteganalyzer()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def quantize_model(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    backend: str = "fbgemm"   # "fbgemm" for x86 CPU, "qnnpack" for ARM (Raspberry Pi)
) -> nn.Module:
    """
    Apply INT8 Post-Training Static Quantization.

    Steps:
      1. Attach qconfig to model (tells PyTorch HOW to quantize each layer).
      2. Insert observers — they record min/max of activations during calibration.
      3. Run calibration pass (feed ~100-200 real images).
      4. Convert: replace float ops with quantized INT8 ops.

    Args:
        model      : FP32 model to quantize
        calib_loader: DataLoader with calibration images (clean, no label needed)
        device     : CPU device (quantization only works on CPU)
        backend    : "fbgemm" (Intel x86) or "qnnpack" (ARM/mobile)

    Returns:
        INT8 quantized model (CPU only)
    """
    model = model.to("cpu")
    model.eval()

    # Step 1: Set qconfig
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    print(f"[Quantize] Using backend: {backend}")
    print(f"[Quantize] qconfig: {model.qconfig}")

    # Step 2: Prepare (inserts observers)
    model_prepared = torch.quantization.prepare(model)
    print("[Quantize] Observers inserted. Running calibration pass...")

    # Step 3: Calibration pass — feed real images
    with torch.no_grad():
        for batch_idx, images in enumerate(calib_loader):
            images = images.to("cpu")
            model_prepared(images)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Calibration: {batch_idx + 1}/{len(calib_loader)} batches done")

    print("[Quantize] Calibration complete.")

    # Step 4: Convert to INT8
    model_quantized = torch.quantization.convert(model_prepared)
    print("[Quantize] Model converted to INT8 ✅")

    return model_quantized


# ─── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_inference(model: nn.Module, image_size: int = 256, n_runs: int = 50) -> float:
    """
    Measure average CPU inference time per image (milliseconds).

    Args:
        model     : FP32 or INT8 model
        image_size: input image size
        n_runs    : number of inference runs for averaging

    Returns:
        Average inference time in milliseconds.
    """
    model.eval().to("cpu")
    dummy = torch.randn(1, 1, image_size, image_size)

    # Warm-up (first run is always slower due to caching)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    print(f"  Inference time: {avg_ms:.1f} ± {std_ms:.1f} ms  (over {n_runs} runs)")
    return avg_ms


def get_model_size_mb(model_path: str) -> float:
    """Return file size of a saved model in MB."""
    return os.path.getsize(model_path) / (1024 * 1024)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="INT8 Quantize the MBCSS fusion model")
    parser.add_argument("--model_path",  default="weights/fusion_best.pt",
                        help="Path to the trained FP32 model (.pt)")
    parser.add_argument("--output_path", default="weights/fusion_int8.pt",
                        help="Where to save the quantized model")
    parser.add_argument("--data_dir",    default="data/clean",
                        help="Directory of calibration images (clean images)")
    parser.add_argument("--n_calib",     type=int, default=200,
                        help="Number of calibration images")
    parser.add_argument("--image_size",  type=int, default=256)
    parser.add_argument("--backend",     default="fbgemm",
                        choices=["fbgemm", "qnnpack"],
                        help="fbgemm = Intel x86 CPU, qnnpack = ARM/mobile")
    args = parser.parse_args()

    device = torch.device("cpu")  # quantization is CPU-only

    # ── Load FP32 model ──────────────────────────────────────────────────────
    print(f"\n[1/4] Loading FP32 model from: {args.model_path}")
    fp32_model = load_model(args.model_path, device)

    # Benchmark FP32
    print("\n[2/4] Benchmarking FP32 model:")
    fp32_time = benchmark_inference(fp32_model, args.image_size)

    # ── Calibration dataset ──────────────────────────────────────────────────
    print(f"\n[3/4] Quantizing (calibrating on {args.n_calib} images)...")
    calib_ds = CalibrationDataset(args.data_dir, n_images=args.n_calib, image_size=args.image_size)
    calib_loader = DataLoader(calib_ds, batch_size=1, shuffle=False)

    int8_model = quantize_model(fp32_model, calib_loader, device, backend=args.backend)

    # ── Save quantized model ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(int8_model.state_dict(), args.output_path)
    print(f"\n[4/4] Quantized model saved to: {args.output_path}")

    # ── Benchmark INT8 ───────────────────────────────────────────────────────
    print("\n[Benchmark] INT8 model:")
    int8_time = benchmark_inference(int8_model, args.image_size)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("QUANTIZATION RESULTS (add these to your paper Table)")
    print("=" * 55)

    fp32_size = get_model_size_mb(args.model_path) if os.path.exists(args.model_path) else 0
    int8_size = get_model_size_mb(args.output_path)

    print(f"  Model size  — FP32: {fp32_size:.1f} MB  →  INT8: {int8_size:.1f} MB"
          f"  ({fp32_size / max(int8_size, 0.001):.1f}× smaller)")
    print(f"  Inference   — FP32: {fp32_time:.1f} ms  →  INT8: {int8_time:.1f} ms"
          f"  ({fp32_time / max(int8_time, 0.001):.1f}× faster)")
    print(f"\n  ⚠ Run evaluate.py on the INT8 model to measure accuracy drop.")
    print(f"    Report the ACTUAL number — don't assume it will be <2%.")
    print("=" * 55)


if __name__ == "__main__":
    main()
