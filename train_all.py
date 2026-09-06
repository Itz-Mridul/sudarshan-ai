#!/usr/bin/env python3
"""
train_all.py
============
Sequential training pipeline — trains ALL branches then fusion automatically.

Run once and walk away:
    python train_all.py

On Google Colab:
    !python train_all.py --epochs 50

On local Mac (MPS):
    python train_all.py --epochs 50

Status is logged to results/training_pipeline.log
"""

import subprocess
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

LOG_FILE = "results/training_pipeline.log"

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs("results", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_stage(name, cmd, required_checkpoint=None):
    """Run one training stage. Returns True if checkpoint was created."""
    log(f"{'='*60}")
    log(f"STAGE: {name}")
    log(f"CMD:   {' '.join(cmd)}")
    log(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"ERROR: {name} exited with code {result.returncode}")
        return False

    # Verify checkpoint exists
    if required_checkpoint and not Path(required_checkpoint).exists():
        log(f"ERROR: Expected checkpoint not found: {required_checkpoint}")
        return False

    log(f"DONE: {name} in {elapsed/60:.1f} min")
    if required_checkpoint:
        size_mb = Path(required_checkpoint).stat().st_size / 1e6
        log(f"Checkpoint: {required_checkpoint} ({size_mb:.1f} MB)")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train all MBCSS branches sequentially")
    parser.add_argument("--epochs",      type=int, default=50, help="Epochs per branch")
    parser.add_argument("--max_images",  type=int, default=None, help="Limit images (quick test)")
    parser.add_argument("--skip_done",   action="store_true", help="Skip stages where checkpoint exists")
    parser.add_argument("--start_from",  choices=["a","b","c","fusion","quantize"], default="a")
    args = parser.parse_args()

    python = sys.executable
    base_cmd = [python, "src/training/train.py"]
    if args.max_images:
        base_cmd += ["--max_images", str(args.max_images)]
    base_cmd += ["--epochs", str(args.epochs)]

    log(f"StegShield — Full Training Pipeline")
    log(f"Epochs per stage: {args.epochs}")
    log(f"Max images: {args.max_images or 'ALL'}")
    log(f"Starting from: branch_{args.start_from}")

    stages = [
        {
            "id": "a",
            "name": "Branch A — Pixel CNN (SRM)",
            "cmd": base_cmd + ["--mode", "branch_a"],
            "checkpoint": "weights/branch_a_best.pt",
            "target": "≥82% val accuracy",
        },
        {
            "id": "b",
            "name": "Branch B — DCT Frequency CNN",
            "cmd": base_cmd + ["--mode", "branch_b"],
            "checkpoint": "weights/branch_b_best.pt",
            "target": "≥76% val accuracy",
        },
        {
            "id": "c",
            "name": "Branch C — Statistical MLP",
            "cmd": base_cmd + ["--mode", "branch_c"],
            "checkpoint": "weights/branch_c_best.pt",
            "target": "≥68% val accuracy",
        },
        {
            "id": "fusion",
            "name": "Fusion — LayerNorm + Dense Classifier",
            "cmd": base_cmd + ["--mode", "fusion"],
            "checkpoint": "weights/fusion_best.pt",
            "target": "≥85-90% val accuracy",
        },
    ]

    order = ["a", "b", "c", "fusion"]
    start_idx = order.index(args.start_from) if args.start_from in order else 0

    results = {}
    for stage in stages[start_idx:]:
        sid = stage["id"]
        ckpt = stage["checkpoint"]

        if args.skip_done and Path(ckpt).exists():
            size_mb = Path(ckpt).stat().st_size / 1e6
            log(f"SKIP: {stage['name']} — checkpoint exists ({size_mb:.1f} MB)")
            results[sid] = "skipped"
            continue

        success = run_stage(stage["name"], stage["cmd"], ckpt)
        results[sid] = "done" if success else "FAILED"

        if not success:
            log(f"Pipeline stopped at stage: {stage['name']}")
            log("Fix the error above, then re-run with --start_from " + sid)
            break

        # After each branch — read and log the best accuracy from history
        hist_path = f"results/history_{stage['name'].split()[0].lower()}.json"
        hist_files = [
            f"results/history_branch_{sid}.json",
            f"results/history_{sid}.json",
        ]
        for hf in hist_files:
            if Path(hf).exists():
                with open(hf) as f:
                    h = json.load(f)
                best_val_acc = max(h.get("val_acc", [0]))
                log(f"Best val accuracy for {stage['name']}: {best_val_acc:.1f}%")
                log(f"Target was: {stage['target']}")
                break

    # ── Quantization ─────────────────────────────────────────────────────────
    if "fusion" in results and results["fusion"] == "done":
        log("\nRunning INT8 quantization...")
        q_cmd = [python, "src/deploy/quantize.py",
                 "--model_path", "weights/fusion_best.pt",
                 "--output_path", "weights/fusion_int8.pt"]
        q_result = subprocess.run(q_cmd, capture_output=False, text=True)
        if q_result.returncode == 0 and Path("weights/fusion_int8.pt").exists():
            orig_mb = Path("weights/fusion_best.pt").stat().st_size / 1e6
            int8_mb = Path("weights/fusion_int8.pt").stat().st_size / 1e6
            log(f"Quantization done: {orig_mb:.1f} MB → {int8_mb:.1f} MB "
                f"({100*(1-int8_mb/orig_mb):.0f}% smaller)")
            results["quantize"] = "done"
        else:
            log("WARNING: Quantization failed — check weights/fusion_best.pt exists")
            results["quantize"] = "FAILED"

    # ── Summary ──────────────────────────────────────────────────────────────
    log("\n" + "="*60)
    log("PIPELINE SUMMARY")
    log("="*60)
    for k, v in results.items():
        icon = "✅" if v in ("done","skipped") else "❌"
        log(f"  {icon}  {k}: {v}")

    all_done = all(v in ("done","skipped") for v in results.values())
    if all_done:
        log("\n🎉 ALL STAGES COMPLETE!")
        log("Next steps:")
        log("  1. python src/training/evaluate.py --ablation")
        log("  2. python src/baselines/xunet_baseline.py --train --epochs 50")
        log("  3. streamlit run app/demo.py")
    else:
        log("\n⚠️  Some stages failed. Check logs above.")

    log(f"\nFull log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
