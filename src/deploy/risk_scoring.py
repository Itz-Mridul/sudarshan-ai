"""
risk_scoring.py
===============
Defense Context Scoring Module (Innovation 2 from PRD)

WHAT THIS DOES:
  Takes the CNN model's stego probability output (0.0 to 1.0) and
  combines it with 4 contextual metadata factors to produce an
  actionable RISK SCORE (0–100) with a severity level.

  This is the "senior inspector" layer — it turns a bare probability
  into intelligence a security officer can act on immediately.

THE FOUR FACTORS:
  1. User privilege   — intern vs employee vs manager
  2. Time of transfer — 2:30 AM transfers are suspicious
  3. File size anomaly — 5 MB where 500 KB is expected → flag
  4. Destination IP   — external IP = highest risk

OUTPUT EXAMPLE:
  "Image 94.7% stego. User: intern. Time: 02:18 AM. Dest: external.
   Risk Score: 87.3 / 100 — CRITICAL → Escalate immediately."

HOW TO USE:
  from src.deploy.risk_scoring import compute_risk_score, RiskLevel

  score_info = compute_risk_score(
      stego_prob=0.947,
      metadata={
          "user_privilege":    "intern",
          "hour":              2,
          "file_size_mb":      5.2,
          "expected_size_mb":  0.5,
          "destination_type":  "external",
      }
  )
  print(score_info["report"])

IMPORTANT — PROTOTYPE STATUS:
  The weights in ScoringWeights are hand-crafted heuristics developed during
  research. They are NOT based on operational security data, red-team studies,
  or calibrated false-positive analysis. This module demonstrates the concept
  and architecture of the risk fusion approach.

  DO NOT use this for autonomous blocking or policy enforcement until:
    1. Weights are calibrated against real operational transfer logs.
    2. False-positive rates are measured and accepted by security officers.
    3. The system is reviewed by a qualified security professional.

  In the demo app, this output is clearly labelled as a PROTOTYPE ESTIMATE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Risk Levels ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Four-tier alert system for security operations (SOC)."""
    LOW      = "LOW"       # score 0–25:  routine, no action
    MEDIUM   = "MEDIUM"    # score 26–50: monitor, log
    HIGH     = "HIGH"      # score 51–75: investigate, notify supervisor
    CRITICAL = "CRITICAL"  # score 76–100: escalate immediately, block transfer


# ─── Score Config ─────────────────────────────────────────────────────────────

@dataclass
class ScoringWeights:
    """
    Tunable weights for each scoring component.
    These map to the patent Claim 2 factors — keep them here in one place
    so they can be cited explicitly in the patent specification.
    """
    # Max contribution from the CNN model (base score)
    model_max:      int = 50   # stego_prob × 50

    # User privilege component (0 to this max)
    privilege_max:  int = 15   # intern=15, employee=7, manager=2

    # Time-of-day anomaly component
    time_max:       int = 15   # off-hours (before 6 AM or after 10 PM) = 15

    # File size anomaly component
    size_max:       int = 10   # ratio > 5× = 10; ratio > 2× = 5

    # Network destination component
    dest_max:       int = 10   # external = 10; DMZ = 5

    # Risk level thresholds (out of 100)
    critical_threshold: int = 75
    high_threshold:     int = 50
    medium_threshold:   int = 25


# ─── Core Scoring Function ────────────────────────────────────────────────────

def compute_risk_score(
    stego_prob: float,
    metadata: dict,
    weights: ScoringWeights = ScoringWeights()
) -> dict:
    """
    Compute the composite defense-context risk score.

    Args:
        stego_prob: float in [0.0, 1.0] — model's stego probability.
                    Use the P(stego) output from F.softmax(logits, dim=1)[:,1].
        metadata:   dict with the following keys (all optional, defaults to lowest risk):
            - user_privilege    (str): "intern" | "employee" | "manager" | "admin"
            - hour              (int): 0–23 — hour of the file transfer event (24h clock)
            - file_size_mb      (float): actual size of the transferred file in MB
            - expected_size_mb  (float): expected size for this image type/resolution
            - destination_type  (str): "internal" | "dmz" | "external"
            - user_id           (str, optional): for the report string
            - filename          (str, optional): for the report string
            - timestamp         (str, optional): human-readable timestamp for report

        weights: ScoringWeights dataclass — tunable per deployment environment.

    Returns:
        dict with keys:
            score       (float): 0.0–100.0 composite risk score
            level       (str):   "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
            breakdown   (dict):  per-component score contributions
            report      (str):   human-readable one-line alert for SOC dashboard
    """
    if not (0.0 <= stego_prob <= 1.0):
        raise ValueError(f"stego_prob must be in [0, 1], got {stego_prob}")

    risk = 0.0
    breakdown = {}

    # ── Component 1: CNN model confidence (max 50) ────────────────────────────
    model_score = stego_prob * weights.model_max
    risk += model_score
    breakdown["model_stego_prob"] = round(model_score, 2)

    # ── Component 2: User privilege (max 15) ──────────────────────────────────
    privilege = metadata.get("user_privilege", "employee").lower()
    privilege_score_map = {
        "intern":   weights.privilege_max,        # 15 — highest risk, least trusted
        "contract": weights.privilege_max - 2,    # 13
        "employee": weights.privilege_max // 2,   # 7
        "manager":  2,
        "admin":    1,                            # admins move large files routinely
    }
    priv_score = privilege_score_map.get(privilege, weights.privilege_max // 2)
    risk += priv_score
    breakdown["user_privilege"] = priv_score

    # ── Component 3: Time anomaly (max 15) ────────────────────────────────────
    hour = int(metadata.get("hour", 12))   # default to noon = lowest risk
    if hour < 6 or hour >= 22:
        time_score = weights.time_max          # off-hours: 0–5 AM, 10 PM–11 PM
    elif hour < 8 or hour >= 20:
        time_score = weights.time_max // 3     # shoulder hours: 5–8 AM, 8–10 PM
    else:
        time_score = 0                          # normal business hours: 8 AM–8 PM
    risk += time_score
    breakdown["time_anomaly"] = time_score

    # ── Component 4: File size anomaly (max 10) ───────────────────────────────
    file_size_mb     = float(metadata.get("file_size_mb", 1.0))
    expected_size_mb = float(metadata.get("expected_size_mb", 1.0))
    if expected_size_mb <= 0:
        expected_size_mb = 0.001          # avoid div-by-zero

    size_ratio = file_size_mb / expected_size_mb
    if size_ratio > 5:
        size_score = weights.size_max          # 5× larger than expected → strong flag
    elif size_ratio > 2:
        size_score = weights.size_max // 2     # 2–5× larger
    elif size_ratio < 0.3:
        size_score = weights.size_max // 4     # unusually small (truncated/corrupted?)
    else:
        size_score = 0                          # within normal range
    risk += size_score
    breakdown["file_size_anomaly"] = size_score

    # ── Component 5: Destination (max 10) ────────────────────────────────────
    dest = metadata.get("destination_type", "internal").lower()
    dest_score_map = {
        "external": weights.dest_max,      # 10 — outside org = highest risk
        "dmz":      weights.dest_max // 2, # 5  — demilitarized zone
        "partner":  3,                      # 3  — trusted partner network
        "internal": 0,                      # 0  — internal transfer, lowest risk
    }
    dest_score = dest_score_map.get(dest, weights.dest_max // 2)
    risk += dest_score
    breakdown["destination"] = dest_score

    # ── Clamp and classify ────────────────────────────────────────────────────
    risk = min(100.0, max(0.0, risk))

    if risk > weights.critical_threshold:
        level = RiskLevel.CRITICAL
    elif risk > weights.high_threshold:
        level = RiskLevel.HIGH
    elif risk > weights.medium_threshold:
        level = RiskLevel.MEDIUM
    else:
        level = RiskLevel.LOW

    # ── Build report string for SOC dashboard ─────────────────────────────────
    user_id   = metadata.get("user_id",   "UNKNOWN")
    filename  = metadata.get("filename",  "image.png")
    timestamp = metadata.get("timestamp", f"{hour:02d}:xx")

    action_map = {
        RiskLevel.CRITICAL: "ESCALATE IMMEDIATELY — Block transfer, notify CISO.",
        RiskLevel.HIGH:     "INVESTIGATE — Notify supervisor, pull transfer logs.",
        RiskLevel.MEDIUM:   "MONITOR — Flag for manual review.",
        RiskLevel.LOW:      "ROUTINE — Log and continue.",
    }

    report = (
        f"[{level.value}] File: '{filename}' | "
        f"Stego probability: {stego_prob * 100:.1f}% | "
        f"User: {user_id} ({privilege}) | "
        f"Time: {timestamp} | "
        f"Destination: {dest.upper()} | "
        f"Risk Score: {risk:.1f}/100 — {action_map[level]}"
    )

    return {
        "score":     round(risk, 1),
        "level":     level.value,
        "breakdown": breakdown,
        "report":    report,
    }


# ─── Batch Scoring ────────────────────────────────────────────────────────────

def batch_score(events: list[dict], weights: ScoringWeights = ScoringWeights()) -> list[dict]:
    """
    Score a list of transfer events in one call.
    Each event dict must have 'stego_prob' and 'metadata' keys.

    Example:
        events = [
            {"stego_prob": 0.94, "metadata": {"user_privilege": "intern", "hour": 2, ...}},
            {"stego_prob": 0.12, "metadata": {"user_privilege": "manager", "hour": 10, ...}},
        ]
        results = batch_score(events)
    """
    return [
        compute_risk_score(event["stego_prob"], event["metadata"], weights)
        for event in events
    ]


# ─── Quick Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Defense Context Risk Scoring — Demo\n")

    test_cases = [
        {
            "label": "CRITICAL case (intern, 2 AM, external, suspicious size)",
            "stego_prob": 0.947,
            "metadata": {
                "user_privilege":   "intern",
                "hour":             2,
                "file_size_mb":     5.2,
                "expected_size_mb": 0.5,
                "destination_type": "external",
                "user_id":          "USR-4472",
                "filename":         "cricket_match.jpg",
                "timestamp":        "02:18 AM",
            }
        },
        {
            "label": "LOW case (manager, 10 AM, internal, normal size)",
            "stego_prob": 0.08,
            "metadata": {
                "user_privilege":   "manager",
                "hour":             10,
                "file_size_mb":     1.1,
                "expected_size_mb": 1.0,
                "destination_type": "internal",
                "user_id":          "MGR-0012",
                "filename":         "site_photo.png",
                "timestamp":        "10:30 AM",
            }
        },
        {
            "label": "MEDIUM case (employee, evening, internal, model uncertain)",
            "stego_prob": 0.55,
            "metadata": {
                "user_privilege":   "employee",
                "hour":             19,
                "file_size_mb":     1.3,
                "expected_size_mb": 1.0,
                "destination_type": "internal",
                "user_id":          "EMP-2231",
                "filename":         "report_scan.png",
                "timestamp":        "07:15 PM",
            }
        },
    ]

    for tc in test_cases:
        print(f"--- {tc['label']} ---")
        result = compute_risk_score(tc["stego_prob"], tc["metadata"])
        print(f"Score: {result['score']}/100  Level: {result['level']}")
        print(f"Report: {result['report']}")
        print(f"Breakdown: {result['breakdown']}")
        print()
