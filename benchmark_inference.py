#!/usr/bin/env python3
"""
benchmark_inference.py
======================
Measures CPU inference latency for all models (for paper Table 3).

Run after quantization:
  python benchmark_inference.py

Outputs:
  results/benchmark_results.json  — latency numbers for paper
  Console table                   — copy directly into paper
"""

import os, sys, time, json, statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

ROOT    = Path(__file__).parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from training.config import Config
from branches.branch_a_pixel import BranchAClassifier
from branches.branch_b_dct import BranchBClassifier
from branches.branch_c_stats import BranchCClassifier
from model.fusion_model import MultiBranchSteganalyzer

# Always benchmark on CPU (deployment target)
DEVICE = torch.device("cpu")
N_WARMUP = 20       # warm-up runs (not counted)
N_RUNS   = 100      # measured runs


def make_dummy_input():
    """Single 256×256 grayscale image batch."""
    return torch.randn(1, 1, 256, 256, device=DEVICE)


def benchmark_model(model, name: str, n_warmup: int = N_WARMUP, n_runs: int = N_RUNS):
    """Measure median CPU latency in milliseconds."""
    model.to(DEVICE)
    model.eval()
    x = make_dummy_input()

    print(f"  Benchmarking {name}...")

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    median_ms = statistics.median(times)
    p95_ms    = sorted(times)[int(0.95 * len(times))]
    return {"median_ms": round(median_ms, 1), "p95_ms": round(p95_ms, 1)}


def get_model_size_mb(path: str) -> float:
    return Path(path).stat().st_size / 1e6


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    cfg = Config()
    results = {}

    models_to_benchmark = [
        {
            "name":  "Branch A (Pixel CNN)",
            "ckpt":  "weights/branch_a_best.pt",
            "mode":  "branch_a",
            "class": lambda: BranchAClassifier(
                feature_dim=cfg.feature_dim_a, num_classes=cfg.num_classes),
        },
        {
            "name":  "Branch B (DCT CNN)",
            "ckpt":  "weights/branch_b_best.pt",
            "mode":  "branch_b",
            "class": lambda: BranchBClassifier(
                feature_dim=cfg.feature_dim_b, num_classes=cfg.num_classes),
        },
        {
            "name":  "Branch C (Stats MLP)",
            "ckpt":  "weights/branch_c_best.pt",
            "mode":  "branch_c",
            "class": lambda: BranchCClassifier(
                feature_dim=cfg.feature_dim_c, num_classes=cfg.num_classes),
        },
        {
            "name":  "MBCSS Fusion FP32",
            "ckpt":  "weights/fusion_best.pt",
            "mode":  "fusion",
            "class": lambda: MultiBranchSteganalyzer(
                feature_dim_a=cfg.feature_dim_a,
                feature_dim_b=cfg.feature_dim_b,
                feature_dim_c=cfg.feature_dim_c,
            ),
        },
        {
            "name":  "MBCSS Fusion INT8",
            "ckpt":  "weights/fusion_int8.pt",
            "mode":  "fusion",
            "class": None,      # quantized — loaded differently
        },
    ]

    print("=" * 65)
    print("  Inference Latency Benchmark (CPU only)")
    print(f"  Warm-up: {N_WARMUP} runs | Measured: {N_RUNS} runs")
    print("=" * 65)

    for entry in models_to_benchmark:
        ckpt = entry["ckpt"]
        name = entry["name"]

        if not Path(ckpt).exists():
            print(f"  ⏳ {name}: {ckpt} not found — skipping")
            continue

        size_mb = get_model_size_mb(ckpt)

        if entry["class"] is not None:
            model = entry["class"]()
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
        else:
            # Quantized model — use torch.load directly
            model = torch.load(ckpt, map_location="cpu")

        n_params = count_params(model)
        lat = benchmark_model(model, name)

        results[name] = {
            "size_mb":   round(size_mb, 2),
            "params":    n_params,
            "median_ms": lat["median_ms"],
            "p95_ms":    lat["p95_ms"],
        }

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  {'Model':<30} {'Size':>8} {'Params':>9} {'Latency':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"  {name:<30} {r['size_mb']:>6.1f}MB {r['params']:>9,} "
              f"  {r['median_ms']:>5.1f}ms (p95={r['p95_ms']:.0f}ms)")
    print("=" * 65)

    # Compute compression ratio if both FP32 and INT8 present
    if "MBCSS Fusion FP32" in results and "MBCSS Fusion INT8" in results:
        fp32 = results["MBCSS Fusion FP32"]
        int8 = results["MBCSS Fusion INT8"]
        size_ratio = fp32["size_mb"] / int8["size_mb"]
        speed_ratio = fp32["median_ms"] / int8["median_ms"]
        print(f"\n  INT8 compression: {size_ratio:.1f}× smaller, {speed_ratio:.1f}× faster")

    # Save
    out = ROOT / "results" / "benchmark_results.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved → {out}")
    print("   Copy these numbers into paper Table 3 (Quantization Results)")


if __name__ == "__main__":
    main()
