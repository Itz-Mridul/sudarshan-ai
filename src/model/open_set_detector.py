"""
open_set_detector.py
====================
Open-Set Steganography Detection for Sudarshana

WHAT THIS DOES:
  Standard classifiers are "closed-set" — they MUST output either "Clean"
  or "Stego", even when they encounter a completely unknown attack method.

  This module adds a third output: "UNKNOWN / UNCERTAIN".

  If the model encounters a steganographic method it has never seen during
  training, it will flag it as uncertain rather than making a wrong guess.

WHY DRDO/ISRO NEEDS THIS:
  In defense scenarios, encountering an unknown threat is common. A model
  that confidently misclassifies an unknown attack as "Clean" is dangerous.
  Open-Set detection ensures the system alerts operators when it encounters
  something it doesn't understand.

HOW IT WORKS:
  We use two complementary approaches:

  Method 1: Softmax Thresholding (Simple, Fast)
    - If the maximum softmax probability is below a threshold τ,
      the sample is flagged as "UNKNOWN".
    - τ is calibrated on the VALIDATION set (never the test set).

  Method 2: Energy-Based Open-Set Detection (More Robust)
    - Instead of softmax, we compute the "energy score":
      E(x) = -log Σ exp(f_i(x))
    - Lower energy = more confident prediction (known class)
    - Higher energy = uncertain / unknown class
    - The energy threshold is also set on the validation set.

THRESHOLD SELECTION:
  The threshold τ is selected using the validation set to maximize
  the trade-off between:
    - True Unknown Rate (TUR): how often unseen methods are correctly flagged
    - False Unknown Rate (FUR): how often known methods are incorrectly flagged
    - Known Accuracy: how well the model classifies known methods

USAGE:
  detector = OpenSetDetector(model, threshold_method="energy")
  detector.calibrate(val_loader)      # Set threshold on validation data
  result = detector.predict(image)    # Returns: class, confidence, is_unknown
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class OpenSetDetector:
    """
    Wraps a trained MultiBranchSteganalyzer with open-set detection capability.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold_method: str = "energy",
        device: torch.device = None,
    ):
        """
        Args:
            model: A trained MultiBranchSteganalyzer
            threshold_method: "softmax" or "energy"
            device: torch device (cpu/cuda)
        """
        self.model = model
        self.threshold_method = threshold_method
        self.device = device or torch.device("cpu")

        # Thresholds (to be calibrated on validation data)
        self.threshold = None
        self.calibration_stats = None

        self.model.eval()
        self.model.to(self.device)

    def _compute_softmax_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum softmax probability for each sample.
        Higher = more confident.
        """
        probs = F.softmax(logits, dim=1)
        max_probs, _ = probs.max(dim=1)
        return max_probs

    def _compute_energy_score(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Compute the energy score: E(x) = -T * log Σ exp(f_i(x) / T)
        Lower energy = more confident (known class).
        Higher energy = uncertain (unknown class).

        We negate it so that higher = more confident (consistent with softmax).
        """
        energy = temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy  # Higher = more confident

    def _compute_score(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute confidence score using the selected method."""
        if self.threshold_method == "softmax":
            return self._compute_softmax_score(logits)
        elif self.threshold_method == "energy":
            return self._compute_energy_score(logits)
        else:
            raise ValueError(f"Unknown method: {self.threshold_method}")

    def calibrate(
        self,
        val_loader,
        target_false_unknown_rate: float = 0.05,
    ) -> Dict:
        """
        Calibrate the open-set threshold using VALIDATION data only.
        Never use test data for calibration!

        The threshold is set such that at most `target_false_unknown_rate`
        fraction of known validation samples are incorrectly flagged as unknown.

        Args:
            val_loader: DataLoader with validation data (known methods only)
            target_false_unknown_rate: maximum acceptable false-unknown rate

        Returns:
            Dict with calibration statistics
        """
        self.model.eval()
        all_scores = []
        all_labels = []
        all_correct = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                logits = self.model(images)
                scores = self._compute_score(logits)
                preds = torch.argmax(logits, dim=1)

                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_correct.extend((preds.cpu() == labels).numpy())

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        all_correct = np.array(all_correct)

        # Set threshold: find the score below which `target_false_unknown_rate`
        # of known samples fall. These would be incorrectly flagged as unknown.
        if self.threshold_method == "softmax":
            # For softmax: lower score = more uncertain
            # Set threshold at the percentile that keeps FUR below target
            self.threshold = float(np.percentile(
                all_scores, target_false_unknown_rate * 100
            ))
        elif self.threshold_method == "energy":
            # For energy: lower energy = more uncertain
            self.threshold = float(np.percentile(
                all_scores, target_false_unknown_rate * 100
            ))

        # Compute calibration statistics
        flagged_unknown = all_scores < self.threshold
        self.calibration_stats = {
            "method": self.threshold_method,
            "threshold": self.threshold,
            "num_val_samples": len(all_scores),
            "val_accuracy": float(np.mean(all_correct)),
            "false_unknown_rate": float(np.mean(flagged_unknown)),
            "mean_score": float(np.mean(all_scores)),
            "std_score": float(np.std(all_scores)),
            "min_score": float(np.min(all_scores)),
            "max_score": float(np.max(all_scores)),
        }

        print(f"✅ Open-Set Calibration Complete:")
        print(f"   Method:             {self.threshold_method}")
        print(f"   Threshold:          {self.threshold:.4f}")
        print(f"   Val Accuracy:       {self.calibration_stats['val_accuracy']*100:.1f}%")
        print(f"   False Unknown Rate: {self.calibration_stats['false_unknown_rate']*100:.1f}%")

        return self.calibration_stats

    def predict(self, x: torch.Tensor) -> Dict:
        """
        Run open-set prediction on an input tensor.

        Args:
            x: image tensor (batch, 1, H, W)

        Returns:
            Dict with:
                - 'class': predicted class (0=clean, 1=stego)
                - 'confidence': model's confidence score
                - 'is_unknown': True if the sample is flagged as unknown
                - 'verdict': "CLEAN", "STEGO", or "UNKNOWN"
                - 'stego_prob': probability of stego class
        """
        if self.threshold is None:
            raise RuntimeError(
                "Open-set threshold not calibrated! "
                "Call detector.calibrate(val_loader) first."
            )

        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x)
            scores = self._compute_score(logits)
            probs = F.softmax(logits, dim=1)
            stego_prob = probs[:, 1]
            preds = torch.argmax(logits, dim=1)

        results = []
        for i in range(x.shape[0]):
            score = float(scores[i].cpu())
            is_unknown = score < self.threshold
            pred_class = int(preds[i].cpu())

            if is_unknown:
                verdict = "UNKNOWN"
            elif pred_class == 0:
                verdict = "CLEAN"
            else:
                verdict = "STEGO"

            results.append({
                "class": pred_class,
                "confidence": score,
                "is_unknown": is_unknown,
                "verdict": verdict,
                "stego_prob": float(stego_prob[i].cpu()),
            })

        return results if len(results) > 1 else results[0]

    def evaluate_open_set(
        self,
        known_loader,
        unknown_loader,
    ) -> Dict:
        """
        Evaluate open-set performance with both known and unknown data.

        Reports the three critical metrics required by the IEEE plan:
          1. True Unknown Rate (TUR): unseen methods correctly flagged as unknown
          2. False Unknown Rate (FUR): known methods incorrectly flagged as unknown
          3. Known Accuracy: classification accuracy on known methods

        Args:
            known_loader: DataLoader with known steganographic methods
            unknown_loader: DataLoader with unseen/unknown methods

        Returns:
            Dict with all open-set metrics
        """
        if self.threshold is None:
            raise RuntimeError("Calibrate the detector first!")

        # ── Evaluate on KNOWN data ─────────────────────────────────────────
        known_correct = 0
        known_total = 0
        known_flagged_unknown = 0

        self.model.eval()
        with torch.no_grad():
            for images, labels in known_loader:
                images = images.to(self.device)
                logits = self.model(images)
                scores = self._compute_score(logits)
                preds = torch.argmax(logits, dim=1)

                for i in range(images.shape[0]):
                    known_total += 1
                    if scores[i].item() < self.threshold:
                        known_flagged_unknown += 1
                    elif preds[i].cpu().item() == labels[i].item():
                        known_correct += 1

        # ── Evaluate on UNKNOWN data ───────────────────────────────────────
        unknown_total = 0
        unknown_flagged_correctly = 0

        with torch.no_grad():
            for images, labels in unknown_loader:
                images = images.to(self.device)
                logits = self.model(images)
                scores = self._compute_score(logits)

                for i in range(images.shape[0]):
                    unknown_total += 1
                    if scores[i].item() < self.threshold:
                        unknown_flagged_correctly += 1

        # ── Compute metrics ────────────────────────────────────────────────
        metrics = {
            "threshold": self.threshold,
            "method": self.threshold_method,

            # Metric 1: True Unknown Rate
            "true_unknown_rate": unknown_flagged_correctly / max(unknown_total, 1),
            "unknown_correct": unknown_flagged_correctly,
            "unknown_total": unknown_total,

            # Metric 2: False Unknown Rate
            "false_unknown_rate": known_flagged_unknown / max(known_total, 1),
            "known_flagged_unknown": known_flagged_unknown,
            "known_total": known_total,

            # Metric 3: Known Accuracy
            "known_accuracy": known_correct / max(known_total - known_flagged_unknown, 1),
            "known_correct": known_correct,
        }

        print()
        print("=" * 60)
        print("  OPEN-SET EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Threshold ({self.threshold_method}): {self.threshold:.4f}")
        print()
        print(f"  True Unknown Rate:  {metrics['true_unknown_rate']*100:.1f}%")
        print(f"    ({unknown_flagged_correctly}/{unknown_total} unknown samples correctly flagged)")
        print()
        print(f"  False Unknown Rate: {metrics['false_unknown_rate']*100:.1f}%")
        print(f"    ({known_flagged_unknown}/{known_total} known samples incorrectly flagged)")
        print()
        print(f"  Known Accuracy:     {metrics['known_accuracy']*100:.1f}%")
        print(f"    ({known_correct} correctly classified out of "
              f"{known_total - known_flagged_unknown} non-flagged known samples)")
        print("=" * 60)

        return metrics


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from model.fusion_model import MultiBranchSteganalyzer
    from torch.utils.data import DataLoader, TensorDataset

    print("Testing Open-Set Detector...")
    print()

    # Create a dummy model
    model = MultiBranchSteganalyzer()
    model.eval()

    # Create dummy validation data
    val_images = torch.randn(100, 1, 256, 256)
    val_labels = torch.randint(0, 2, (100,))
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Create dummy unknown data
    unknown_images = torch.randn(50, 1, 256, 256)
    unknown_labels = torch.ones(50, dtype=torch.long)  # all stego
    unknown_dataset = TensorDataset(unknown_images, unknown_labels)
    unknown_loader = DataLoader(unknown_dataset, batch_size=16)

    # Test softmax method
    print("─" * 40)
    print("  Method: Softmax Thresholding")
    print("─" * 40)
    detector_sm = OpenSetDetector(model, threshold_method="softmax")
    detector_sm.calibrate(val_loader, target_false_unknown_rate=0.05)
    metrics_sm = detector_sm.evaluate_open_set(val_loader, unknown_loader)

    print()

    # Test energy method
    print("─" * 40)
    print("  Method: Energy-Based Detection")
    print("─" * 40)
    detector_en = OpenSetDetector(model, threshold_method="energy")
    detector_en.calibrate(val_loader, target_false_unknown_rate=0.05)
    metrics_en = detector_en.evaluate_open_set(val_loader, unknown_loader)

    # Test single prediction
    print()
    single_img = torch.randn(1, 1, 256, 256)
    result = detector_en.predict(single_img)
    print(f"  Single prediction: {result['verdict']} "
          f"(confidence={result['confidence']:.3f}, "
          f"is_unknown={result['is_unknown']})")

    print("\n✅ Open-Set Detector OK!")
