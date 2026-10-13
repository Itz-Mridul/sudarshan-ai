"""Deep-learning evidence adapter for the existing multi-branch model.

The adapter intentionally does not call the model "universal" unless its
checkpoint has actually been trained and evaluated across the target attack
families. It exposes the learned fusion probability plus branch evidence for
the expert dashboard.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def verify_with_blind_cnn(model, image_tensor: torch.Tensor, trained: bool) -> Dict[str, object]:
    """Run the trained multi-branch CNN and return explainable evidence."""
    if not trained:
        return {
            "available": False,
            "model_name": "Multi-branch blind CNN",
            "scope_note": "No trained checkpoint is loaded.",
        }

    features_out = model(image_tensor, return_branch_features=True)
    logits = features_out[0]
    feat_a, feat_b, feat_c = features_out[1], features_out[2], features_out[3]
    probs = F.softmax(logits, dim=1)[0]
    residual_input = image_tensor * 127.5 + 127.5
    residuals = model.branch_a.srm(residual_input)
    residual_energy = float(residuals.abs().mean().cpu())
    return {
        "available": True,
        "model_name": "Multi-branch blind CNN (SRM + DCT + statistics)",
        "stego_probability": float(probs[1].cpu()),
        "clean_probability": float(probs[0].cpu()),
        "branch_feature_norms": {
            "pixel_srm": float(feat_a.norm().cpu()),
            "dct_frequency": float(feat_b.norm().cpu()),
            "statistics": float(feat_c.norm().cpu()),
        },
        "residual_energy": residual_energy,
        "scope_note": "This is blind only within the algorithms represented in training. Cross-algorithm validation is required before a universal claim.",
    }
