"""Regular-Singular (RS) steganalysis for grayscale images.

This is a screening implementation of the classic RS method. It compares the
roughness of small pixel groups before and after a +/-1 LSB mask operation.
The returned score is deliberately described as a screening score: calibration
against a cover source and embedding algorithm is required before treating it
as a probability.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


_DEFAULT_MASK = np.array([0, 1, 1, 0], dtype=np.int8)


def _variation(group: np.ndarray) -> float:
    """Measure local roughness using adjacent absolute differences."""
    return float(np.abs(np.diff(group.astype(np.int16))).sum())


def _shift(group: np.ndarray, mask: np.ndarray, direction: int) -> np.ndarray:
    """Apply a bounded +/-1 shift to the masked pixels."""
    shifted = group.astype(np.int16).copy()
    selected = mask.astype(bool)
    shifted[selected] += direction
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _classify(original: np.ndarray, transformed: np.ndarray) -> tuple[int, int, int]:
    before = _variation(original)
    after = _variation(transformed)
    if after > before:
        return 1, 0, 0  # regular
    if after < before:
        return 0, 1, 0  # singular
    return 0, 0, 1       # unusable/equal


def analyse_rs(image: np.ndarray, group_size: int = 4) -> Dict[str, float | int | str]:
    """Analyse an image with a compact RS implementation.

    Args:
        image: 2-D grayscale ``uint8`` array.
        group_size: number of pixels per group. The default is the classical
            four-pixel group used by many RS implementations.

    Returns:
        Counts, normalized R/S differences, and a 0–1 screening score. The
        score is not a calibrated probability.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("RS steganalysis expects a 2-D grayscale image")
    if arr.size < group_size * 8:
        raise ValueError("RS steganalysis needs at least 32 grayscale pixels")
    if group_size != len(_DEFAULT_MASK):
        raise ValueError("This implementation currently supports group_size=4")

    flat = np.clip(arr, 0, 255).astype(np.uint8).reshape(-1)
    usable = (flat.size // group_size) * group_size
    groups = flat[:usable].reshape(-1, group_size)
    mask = _DEFAULT_MASK

    plus = np.zeros(3, dtype=np.int64)
    minus = np.zeros(3, dtype=np.int64)
    valid_groups = 0
    for group in groups:
        # Boundary clipping makes +1 and -1 transformations asymmetric. Do
        # not score such groups; otherwise a flat black/white image becomes a
        # false positive simply because one direction is clipped.
        if np.any((mask.astype(bool)) & ((group == 0) | (group == 255))):
            continue
        valid_groups += 1
        plus += _classify(group, _shift(group, mask, +1))
        minus += _classify(group, _shift(group, mask, -1))

    if valid_groups < 8:
        return {
            "groups": int(len(groups)),
            "usable_groups": int(valid_groups),
            "regular_plus": 0,
            "singular_plus": 0,
            "regular_minus": 0,
            "singular_minus": 0,
            "delta_plus": 0.0,
            "delta_minus": 0.0,
            "rs_gap": 0.0,
            "screening_score": 0.0,
            "finding": "insufficient unsaturated groups for RS screening",
            "calibrated": False,
        }

    n = float(valid_groups)
    delta_plus = float((plus[0] - plus[1]) / n)
    delta_minus = float((minus[0] - minus[1]) / n)
    gap = abs(delta_plus - delta_minus)
    # This is intentionally a screening normalization, not a learned or
    # statistically calibrated probability. A validation set should determine
    # the operating threshold for a deployment.
    screening_score = float(np.clip(gap / 0.10, 0.0, 1.0))
    if screening_score >= 0.65:
        finding = "elevated RS irregularity"
    elif screening_score >= 0.35:
        finding = "borderline RS irregularity"
    else:
        finding = "no strong RS irregularity"

    return {
        "groups": int(len(groups)),
        "usable_groups": int(valid_groups),
        "regular_plus": int(plus[0]),
        "singular_plus": int(plus[1]),
        "regular_minus": int(minus[0]),
        "singular_minus": int(minus[1]),
        "delta_plus": delta_plus,
        "delta_minus": delta_minus,
        "rs_gap": float(gap),
        "screening_score": screening_score,
        "finding": finding,
        "calibrated": False,
    }
