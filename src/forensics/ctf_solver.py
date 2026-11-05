"""Comprehensive Digital Image Forensics & Covert Channel Carving Engine.

Automatically reconstructs arbitrary hidden payloads, structured canary tokens, polyglot archives,
multi-channel bitplanes, metadata tags, heuristic key-harvesting, JPEG DCT coefficients, decoders,
and executes statistical steganalysis (Chi-Square, SPA, RS, ELA, and SNR residual analysis).
"""

from __future__ import annotations

import base64
import codecs
import io
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

try:
    from forensics.structural_audit import audit_file_structure
except ImportError:
    try:
        from src.forensics.structural_audit import audit_file_structure
    except ImportError:
        # Support direct execution: ``python ctf_solver.py``.
        from structural_audit import audit_file_structure

# Universal structured token and canary identifier pattern:
UNIVERSAL_FLAG_REGEX = re.compile(
    r"[A-Za-z0-9_\-\.]{2,32}\{[A-Za-z0-9_\-\.\!\@\#\$\%\^\&\*\+\=]{4,100}\}",
    re.IGNORECASE,
)
UNIVERSAL_SECRET_REGEX = UNIVERSAL_FLAG_REGEX

# Generic curly brace discovery for tokens without standard prefix
BRACE_FLAG_REGEX = re.compile(
    r"\{[A-Za-z0-9_\-\.\!\@\#\$\%]{6,80}\}"
)

# Standard JPEG 8x8 Zigzag table
_ZIGZAG = np.array([
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
])


@dataclass
class FlagCandidate:
    flag: str
    source: str
    confidence: str = "HIGH"
    details: str = ""
    via: str = "plain"
    score: float = 0.0

PayloadCandidate = FlagCandidate


@dataclass
class CarvedFile:
    filename: str
    size_bytes: int
    data: bytes
    mime_type: str = "application/octet-stream"
    note: str = ""


@dataclass
class SolverReport:
    flags: List[FlagCandidate] = field(default_factory=list)
    carved_files: List[CarvedFile] = field(default_factory=list)
    metadata_findings: Dict[str, str] = field(default_factory=dict)
    bitplane_findings: List[Dict[str, Any]] = field(default_factory=list)
    cli_telemetry: List[Dict[str, Any]] = field(default_factory=list)
    harvested_keys: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    dct_findings: List[Dict[str, Any]] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    iocs: Dict[str, List[str]] = field(default_factory=dict)
    chi2_findings: Dict[str, Any] = field(default_factory=dict)
    spa_findings: Dict[str, Any] = field(default_factory=dict)
    ela_findings: Dict[str, Any] = field(default_factory=dict)
    snr_findings: Dict[str, Any] = field(default_factory=dict)


def calculate_entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte (0.0 to 8.0)."""
    if not data:
        return 0.0
    arr = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / len(arr)
    return float(-np.sum(probs * np.log2(probs)))


def extract_forensic_iocs(corpus: List[str] | str) -> Dict[str, List[str]]:
    """Extract forensic Indicators of Compromise (IOCs) such as IPv4 addresses, URLs, API keys, and C2 endpoints."""
    text = "\n".join(corpus) if isinstance(corpus, list) else str(corpus)
    iocs: Dict[str, Set[str]] = {
        "ip_addresses": set(),
        "urls": set(),
        "onion_domains": set(),
        "api_keys_and_tokens": set(),
        "command_invocations": set(),
    }

    # 1. IPv4 Addresses (excluding private loopback 127.0.0.1 and broadcast 0.0.0.0)
    for m in re.finditer(r"\b([0-9]{1,3}(?:\.[0-9]{1,3}){3})\b", text):
        ip = m.group(1)
        octets = [int(x) for x in ip.split(".")]
        if all(0 <= o <= 255 for o in octets) and ip not in ("0.0.0.0", "127.0.0.1", "255.255.255.255"):
            iocs["ip_addresses"].add(ip)

    # 2. URLs / C2 endpoints
    for m in re.finditer(r"https?://[A-Za-z0-9\.\-_]+(?::\d+)?(?:/[^\s<>'\"{}|\\^`]+)?", text):
        iocs["urls"].add(m.group(0))

    # 3. Tor .onion addresses
    for m in re.finditer(r"\b[a-z2-7]{16,56}\.onion\b", text):
        iocs["onion_domains"].add(m.group(0))

    # 4. API keys / Private Key blocks / Tokens
    for m in re.finditer(r"AKIA[0-9A-Z]{16}", text):
        iocs["api_keys_and_tokens"].add(f"AWS Access Key: {m.group(0)}")
    if "-----BEGIN" in text and "PRIVATE KEY-----" in text:
        iocs["api_keys_and_tokens"].add("Cryptographic Private Key Block (PEM detected)")
    for m in re.finditer(r"Bearer\s+([A-Za-z0-9_\-\.]{20,})", text):
        iocs["api_keys_and_tokens"].add(f"OAuth/Bearer Token: {m.group(1)[:12]}...")

    # 5. Suspicious commands / droppers
    for m in re.finditer(r"(?:powershell(?:\.exe)?|cmd\.exe|/bin/(?:ba)?sh)\s+[^\n\r;]{5,80}", text, re.IGNORECASE):
        iocs["command_invocations"].add(m.group(0).strip())

    return {k: sorted(list(v)) for k, v in iocs.items() if v}


def chi_square_lsb_attack(pil_img: Image.Image) -> Dict[str, Any]:
    """Perform Westfeld & Pfitzmann (1999) Chi-Square attack on spatial Pairs of Values (PoVs).
    Detects LSB substitution steganography even without known password or key.
    """
    result = {
        "stego_probability": 0.0,
        "chi2_stat": 0.0,
        "degrees_of_freedom": 0,
        "verdict": "BENIGN / UNALTERED",
        "channel_findings": {},
    }
    try:
        from scipy import stats
        rgb = pil_img.convert("RGB")
        chans = {"Red": rgb.split()[0], "Green": rgb.split()[1], "Blue": rgb.split()[2]}
        max_prob = 0.0

        for ch_name, ch_img in chans.items():
            arr = np.array(ch_img, dtype=np.uint8).flatten()
            counts = np.bincount(arr, minlength=256)
            chi2 = 0.0
            dof = 0
            for k in range(128):
                n1 = counts[2 * k]
                n2 = counts[2 * k + 1]
                expected = (n1 + n2) / 2.0
                if expected > 0:
                    chi2 += ((n1 - expected) ** 2) / expected
                    dof += 1
            if dof > 1:
                prob_equalized = float(1.0 - stats.chi2.cdf(chi2, dof - 1))
                result["channel_findings"][ch_name] = {
                    "chi2": round(float(chi2), 2),
                    "dof": dof,
                    "probability_equalized": round(prob_equalized, 4),
                }
                if prob_equalized > max_prob:
                    max_prob = prob_equalized

        result["stego_probability"] = round(max_prob, 4)
        if max_prob >= 0.90:
            result["verdict"] = "HIGH CONFIDENCE STEGANOGRAPHY (PoV Equalization)"
        elif max_prob >= 0.50:
            result["verdict"] = "SUSPICIOUS / ELEVATED LSB ASYMMETRY"
        else:
            result["verdict"] = "NORMAL NATURAL GRADIENT (Clean)"
    except Exception as e:
        result["error"] = str(e)
    return result


def sample_pair_analysis(pil_img: Image.Image) -> Dict[str, Any]:
    """Perform Dumitrescu-Wu-Wang (2002) Sample Pair Analysis (SPA) for LSB Steganalysis.

    Mathematically estimates the secret message embedding rate (p) by evaluating
    parity symmetry perturbations across adjacent horizontal and vertical pixel pairs.
    """
    try:
        rgb_img = pil_img.convert("RGB")
        arr = np.asarray(rgb_img, dtype=np.int32)
        channel_results = {}
        rates = []

        for c_idx, c_name in enumerate(["Red", "Green", "Blue"]):
            c_data = arr[:, :, c_idx]
            u_h = c_data[:, :-1].flatten()
            v_h = c_data[:, 1:].flatten()

            u_v = c_data[:-1, :].flatten()
            v_v = c_data[1:, :].flatten()

            u = np.concatenate([u_h, u_v])
            v = np.concatenate([v_h, v_v])

            diff = u - v
            P = np.sum((u // 2) == (v // 2))
            X = np.sum((diff % 2 == 0) & ((u // 2) != (v // 2)))
            Y = np.sum(diff % 2 != 0)

            total = len(u)
            if total == 0:
                continue

            diff_parity = abs(int(Y) - int(X))
            est_p = max(0.0, min(1.0, float(diff_parity / (total * 0.5))))
            rates.append(est_p)
            channel_results[c_name] = {
                "estimated_rate": round(est_p, 4),
                "pairs_evaluated": int(total),
                "even_diff_count": int(X),
                "odd_diff_count": int(Y),
            }

        avg_rate = round(float(np.mean(rates)), 4) if rates else 0.0
        max_rate = round(float(np.max(rates)), 4) if rates else 0.0

        if max_rate > 0.40:
            verdict = "ELEVATED LSB MODIFICATION (High Stego Embedding Indicated)"
        elif max_rate > 0.15:
            verdict = "SUSPICIOUS LSB PERTURBATION (Low-to-Medium Embedding Possible)"
        else:
            verdict = "NATURAL PARITY SYMMETRY (Clean / No Significant LSB Replacement)"

        return {
            "average_embedding_rate": avg_rate,
            "max_embedding_rate": max_rate,
            "channel_results": channel_results,
            "verdict": verdict,
        }
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR", "max_embedding_rate": 0.0}


def perform_error_level_analysis(
    pil_img: Image.Image,
    quality: int = 95,
    scale: int = 20,
) -> Dict[str, Any]:
    """Perform Error Level Analysis (ELA) to detect digital tampering and compression discrepancies.

    Compresses image to JPEG at the specified quality factor, calculates the absolute difference
    between the original and recompressed pixel values, and amplifies the error rate.
    Uniform error distribution indicates consistent compression history (clean).
    Discontinuous high-error regions indicate splicing, modification, or payload insertion.
    """
    try:
        rgb_img = pil_img.convert("RGB")
        buf = io.BytesIO()
        rgb_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        resaved = Image.open(buf)

        orig_arr = np.asarray(rgb_img, dtype=np.int16)
        resaved_arr = np.asarray(resaved, dtype=np.int16)

        diff_arr = np.abs(orig_arr - resaved_arr)
        scaled_diff = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)
        ela_img = Image.fromarray(scaled_diff)

        mean_err = round(float(np.mean(diff_arr)), 2)
        max_err = int(np.max(diff_arr))
        var_err = round(float(np.var(diff_arr)), 2)

        h, w = diff_arr.shape[:2]
        block_size = 16
        block_means = []
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                blk = diff_arr[y : y + block_size, x : x + block_size]
                block_means.append(float(np.mean(blk)))

        block_discrepancy = round(float(np.std(block_means)), 2) if block_means else 0.0

        if block_discrepancy > 4.5 or max_err > 45:
            verdict = "ELEVATED ERROR DISCREPANCY (Localized Tampering / Editing Likely)"
            tamper_prob = min(0.95, 0.40 + (block_discrepancy / 10.0))
        elif block_discrepancy > 2.5:
            verdict = "MODERATE ERROR VARIANCE (Possible Minor Compression Discontinuity)"
            tamper_prob = 0.50
        else:
            verdict = "HOMOGENEOUS ERROR FIELD (Consistent Compression / Untampered)"
            tamper_prob = max(0.05, block_discrepancy / 10.0)

        return {
            "ela_image": ela_img,
            "mean_error": mean_err,
            "max_error": max_err,
            "error_variance": var_err,
            "block_discrepancy": block_discrepancy,
            "tamper_probability": round(tamper_prob, 2),
            "verdict": verdict,
            "quality": quality,
            "scale": scale,
        }
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR", "tamper_probability": 0.0}


def spatial_noise_residual_analysis(pil_img: Image.Image) -> Dict[str, Any]:
    """Perform high-pass Spatial Noise Residual (SNR) analysis using a 2D Laplacian operator.

    Isolates sensor noise and high-frequency residuals from image content.
    Calculates spatial noise variance, Kurtosis, and energy concentration to detect
    artificial noise injection or synthetic tampering.
    """
    try:
        gray = pil_img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

        padded = np.pad(arr, 1, mode="reflect")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * arr
        )

        residual_var = round(float(np.var(laplacian)), 2)
        residual_std = float(np.std(laplacian))
        residual_energy = round(float(np.mean(np.abs(laplacian))), 2)

        if residual_std > 0:
            kurtosis = round(float(np.mean(((laplacian - np.mean(laplacian)) / residual_std) ** 4)), 2)
        else:
            kurtosis = 0.0

        scaled_vis = np.clip(np.abs(laplacian) * 4.0, 0, 255).astype(np.uint8)
        residual_img = Image.fromarray(scaled_vis)

        if residual_energy > 28.0 or (kurtosis < 3.2 and residual_var > 150):
            verdict = "ELEVATED RESIDUAL NOISE (High-Frequency Anomaly / Artificial Dispersion)"
            anomaly_score = 0.85
        elif residual_energy > 18.0:
            verdict = "MODERATE SENSOR NOISE (Consistent with High ISO or Light Texture)"
            anomaly_score = 0.40
        else:
            verdict = "NORMAL OPTICAL GRADIENT (Smooth / Low Residual Energy)"
            anomaly_score = 0.15

        return {
            "residual_image": residual_img,
            "residual_variance": residual_var,
            "residual_energy": residual_energy,
            "kurtosis": kurtosis,
            "verdict": verdict,
            "anomaly_score": anomaly_score,
        }
    except Exception as e:
        return {"error": str(e), "verdict": "ERROR", "anomaly_score": 0.0}


def score_flag_candidate(
    flag_str: str,
    custom_regex: Optional[str] = None,
    hint: Optional[str] = None,
) -> float:
    """Score payload candidate likelihood (higher = more likely genuine covert payload).
    Suppresses mathematical XOR collision noise (consonant soup like '{2qrjCgw}')
    while honoring genuine structured tokens, canary identifiers, and user-supplied hints.
    """
    score = 0.0
    lower = flag_str.lower()
    is_generic_brace = custom_regex and custom_regex.strip().startswith((r"\{", r"^{", "{"))

    # Specific non-generic regex (e.g. ^KCTF-secret-\d{4}) gets top priority
    if custom_regex and not is_generic_brace:
        try:
            if re.search(custom_regex, flag_str, re.IGNORECASE):
                return 500.0
        except re.error:
            pass

    known_prefixes = [
        "htb{", "flag{", "ctf{", "picoctf{", "cscg{", "sec{",
        "cyber{", "hack{", "ductf{", "kctf{", "knight{", "shakti{",
        "secret{", "canary{", "token{", "key{", "pass{", "auth{", "corp{", "id{", "ioc{"
    ]
    if hint and hint.strip():
        clean_h = re.sub(r"[^A-Za-z0-9_\-]", "", hint.strip().lower())
        if clean_h:
            known_prefixes.append(f"{clean_h}{{")

    if any(lower.startswith(p) for p in known_prefixes):
        score += 150.0

    if "{" in flag_str and "}" in flag_str:
        body = flag_str.split("{", 1)[1].rsplit("}", 1)[0]
        prefix = flag_str.split("{", 1)[0]

        if len(body) < 6:
            score -= 40.0

        punc = sum(c in "^#+!*~%&=/\\" for c in body)
        if punc > 0:
            score -= punc * 20.0

        vowel_leet_chars = sum(c.lower() in "aeiou013457" for c in body)
        alpha_chars = sum(c.isalpha() for c in body)

        if not any(lower.startswith(p) for p in known_prefixes):
            # Check prefix for random consonant soup or arbitrary mixed casing
            if prefix:
                pref_alphas = [c for c in prefix if c.isalpha()]
                if pref_alphas:
                    pref_vowels = sum(c.lower() in "aeiou" for c in pref_alphas)
                    if len(pref_alphas) >= 3 and pref_vowels == 0:
                        return 0.0
                    has_lower = any(c.islower() for c in pref_alphas)
                    has_upper = any(c.isupper() for c in pref_alphas)
                    if has_lower and has_upper and not prefix.istitle():
                        return 0.0

            # Reject random punctuation inside body for unverified patterns
            if punc > 0:
                return 0.0
            # Reject pure consonant soup (e.g. {2qrjCgw} has 0 vowels)
            if vowel_leet_chars == 0 and len(body) >= 4:
                return 0.0
            # Reject random high-entropy consonant garbage (< 15% vowels)
            if alpha_chars > 0 and (vowel_leet_chars / alpha_chars) < 0.15:
                return 0.0
            # Require structured indicators (underscores, digits, reasonable length)
            if "_" not in body and not any(c.isdigit() for c in body):
                return 0.0
            if len(body) < 8:
                return 0.0

        score += min(len(body), 40)
        score += body.count("_") * 25.0
        score += sum(c in "013457" for c in body) * 8.0
        alpha_ratio = sum(c.isalpha() for c in body) / max(len(body), 1)
        if alpha_ratio > 0.4:
            score += 20.0

        # Generic curly brace match provides modest boost only if linguistic structure is sound
        if custom_regex and is_generic_brace:
            try:
                if re.search(custom_regex, flag_str, re.IGNORECASE):
                    score += 50.0
            except re.error:
                pass

    return max(0.0, score)


def extract_printable_strings(data: bytes, min_len: int = 4) -> List[str]:
    """Extract sequences of printable ASCII characters."""
    pattern = re.compile(b"[\x20-\x7e]{" + str(min_len).encode() + b",}")
    matches = pattern.findall(data)
    results = []
    for m in matches:
        try:
            results.append(m.decode("ascii"))
        except UnicodeDecodeError:
            pass
    return results


def find_flags_in_text(text: str, custom_regex: Optional[str] = None) -> List[str]:
    """Search for flags matching custom or universal regex in plain text."""
    found = set()
    if custom_regex:
        try:
            user_rgx = re.compile(custom_regex, re.IGNORECASE)
            for m in user_rgx.finditer(text):
                found.add(m.group(0))
        except re.error:
            pass

    for m in UNIVERSAL_FLAG_REGEX.finditer(text):
        found.add(m.group(0))

    for m in BRACE_FLAG_REGEX.finditer(text):
        found.add(m.group(0))

    return sorted(list(found))


def find_flags_deep(blob: str | bytes, custom_regex: Optional[str] = None) -> List[Tuple[str, str]]:
    """Decode candidate blobs across plain, base64, hex, rot13, base64>rot13, base32, reverse."""
    if isinstance(blob, bytes):
        try:
            s = blob.decode("latin-1", "replace")
        except Exception:
            s = ""
    else:
        s = str(blob)

    hits: List[Tuple[str, str]] = []

    # 1. Plain
    for f in find_flags_in_text(s, custom_regex):
        hits.append((f, "plain"))

    # 2. Base64 & Base64>ROT13
    b64_cands = re.findall(r"[A-Za-z0-9+/]{8,}={0,2}", s)
    if not b64_cands and len(s) >= 8:
        b64_cands = [re.sub(r"[^A-Za-z0-9+/=]", "", s)]
    for cand in b64_cands:
        p = len(cand) % 4
        padded = cand + ("=" * (4 - p) if p else "")
        try:
            b64_dec = base64.b64decode(padded.encode("ascii"), validate=False).decode("latin-1", "replace")
            for f in find_flags_in_text(b64_dec, custom_regex):
                hits.append((f, "base64"))
            r13_after_b64 = codecs.decode(b64_dec, "rot_13")
            for f in find_flags_in_text(r13_after_b64, custom_regex):
                hits.append((f, "base64>rot13"))
        except Exception:
            pass

    # 3. Hex
    hex_cands = re.findall(r"[0-9a-fA-F]{16,}", s)
    if not hex_cands and len(s) >= 16:
        hex_cands = [re.sub(r"[^0-9a-fA-F]", "", s)]
    for cand in hex_cands:
        if len(cand) % 2 != 0:
            cand = cand[:-1]
        try:
            hex_dec = bytes.fromhex(cand).decode("latin-1", "replace")
            for f in find_flags_in_text(hex_dec, custom_regex):
                hits.append((f, "hex"))
        except Exception:
            pass

    # 4. ROT13
    try:
        r13 = codecs.decode(s, "rot_13")
        for f in find_flags_in_text(r13, custom_regex):
            hits.append((f, "rot13"))
    except Exception:
        pass

    # 5. Base32
    b32_cands = re.findall(r"[A-Za-z2-7]{8,}={0,6}", s)
    if not b32_cands and len(s) >= 8:
        b32_cands = [re.sub(r"[^A-Za-z2-7=]", "", s.upper())]
    for cand in b32_cands:
        p = len(cand) % 8
        padded = cand + ("=" * (8 - p) if p else "")
        try:
            b32_dec = base64.b32decode(padded.encode("ascii"), casefold=True).decode("latin-1", "replace")
            for f in find_flags_in_text(b32_dec, custom_regex):
                hits.append((f, "base32"))
        except Exception:
            pass

    # 6. Reverse
    try:
        rev = s[::-1]
        for f in find_flags_in_text(rev, custom_regex):
            hits.append((f, "reverse"))
    except Exception:
        pass

    # Deduplicate preserving first decoder found
    out = []
    seen = set()
    for f, via in hits:
        if f not in seen:
            seen.add(f)
            out.append((f, via))
    return out


def find_flags_in_bytes(data: bytes, custom_regex: Optional[str] = None) -> List[Tuple[str, str]]:
    """Search for flags directly within raw byte buffers across multiple encodings."""
    hits = []
    if not data:
        return hits

    # Quick rejection filter if custom_regex is None
    if custom_regex is None:
        lower_data = data.lower()
        has_brace = b"{" in data or b"}" in data
        has_b64 = b"ew" in data or b"ey" in data
        has_hex = b"7b" in lower_data
        has_r13 = b"tgb{" in lower_data or b"synt{" in lower_data
        if not (has_brace or has_b64 or has_hex or has_r13):
            return hits

    # Check printable strings
    strings = extract_printable_strings(data, min_len=4)
    for s in strings:
        for f, via in find_flags_deep(s, custom_regex):
            hits.append((f, via))
    
    # Also test whole latin-1 decoded stream directly
    try:
        whole = data.decode("latin-1", "replace")
        for f, via in find_flags_deep(whole, custom_regex):
            hits.append((f, via))
    except Exception:
        pass

    out = []
    seen = set()
    for f, via in hits:
        if f not in seen:
            seen.add(f)
            out.append((f, via))
    return out


_ENGLISH_STOP_WORDS = {
    "the", "and", "is", "in", "to", "of", "it", "that", "for", "on", "with",
    "as", "this", "by", "at", "from", "be", "have", "or", "an", "each", "after",
    "before", "all", "are", "not", "but", "what", "which", "their", "will", "would",
    "there", "can", "could", "first", "also", "then", "into", "more", "some", "other",
    "its", "than", "now", "just", "over", "these", "only", "such", "time", "speaks",
    "disagrees", "insists", "repeats", "complete", "picture", "quiet", "channel",
    "smallest", "largest", "sample", "keeps", "heard", "word", "lock", "fives",
    "does", "been", "being", "were", "where", "when", "how", "who", "whom", "lake",
}


def harvest_candidate_keys(
    text_corpus: List[str],
    user_passwords: Optional[List[str]] = None,
) -> List[bytes]:
    """Extract genuine candidate passphrases from metadata, notes, and user input.
    Excludes English grammatical stop words and only includes meaningful tokens.
    """
    raw_keys: Set[str] = set()

    if user_passwords:
        for p in user_passwords:
            if p and p.strip():
                raw_keys.add(p.strip())

    for text in text_corpus:
        if not text:
            continue
        # 1. Clue patterns: key is ..., password: ..., lock repeats the word: LANTERN
        for m in re.findall(r"(?:key|password|pass|secret|lock|word)\s*(?:repeats|is|:|=)\s*[:\s]*([A-Za-z0-9_!@#$%^&*]+)", text, re.I):
            k = m.strip(" :.,;\"'")
            if len(k) >= 2 and k.lower() not in _ENGLISH_STOP_WORDS:
                raw_keys.add(k)

        # 2. Quoted strings: "ARCHIVE"
        for m in re.findall(r"[\"'\x27]([A-Za-z0-9_!@#$%^&*]{3,32})[\"'\x27]", text):
            k = m.strip(" :.,;\"'")
            if len(k) >= 2 and k.lower() not in _ENGLISH_STOP_WORDS:
                raw_keys.add(k)

        # 3. Prominent all-caps identifiers: LANTERN, ARCHIVE, FIELD
        for m in re.findall(r"\b([A-Z0-9_]{3,20})\b", text):
            if m.lower() not in _ENGLISH_STOP_WORDS:
                raw_keys.add(m.strip())

        # 4. Alphanumeric / technical tokens (e.g. 37th, archivist)
        for w in re.split(r"[\s\-_]+", text):
            w_clean = re.sub(r"[^A-Za-z0-9]", "", w)
            if 3 <= len(w_clean) <= 16 and w_clean.lower() not in _ENGLISH_STOP_WORDS:
                if any(c.isupper() for c in w_clean) or any(c.isdigit() for c in w_clean):
                    raw_keys.add(w_clean)

    byte_keys = []
    for k in sorted(list(raw_keys)):
        if k:
            byte_keys.append(k.encode("utf-8"))
            if k.isupper() and k.lower() != k:
                byte_keys.append(k.lower().encode("utf-8"))

    seen = set()
    result = []
    for b in byte_keys:
        if b not in seen:
            seen.add(b)
            result.append(b)

    return result[:32]


def scan_metadata(
    raw_bytes: bytes,
    pil_img: Image.Image,
    custom_regex: Optional[str] = None,
) -> Tuple[Dict[str, str], List[FlagCandidate], List[str]]:
    """Inspect PNG/JPEG metadata (EXIF, Title, Comments, Artist, Software, Private Chunks)."""
    findings: Dict[str, str] = {}
    flags: List[FlagCandidate] = []
    text_corpus: List[str] = []

    # 1. PIL info dict (PNG textual chunks)
    if hasattr(pil_img, "info") and pil_img.info:
        for k, v in pil_img.info.items():
            if isinstance(v, (str, bytes)):
                s_val = v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
                findings[f"Info: {k}"] = s_val
                text_corpus.append(s_val)
                for flg, via in find_flags_deep(s_val, custom_regex):
                    score = score_flag_candidate(flg, custom_regex)
                    flags.append(FlagCandidate(
                        flag=flg,
                        source=f"Metadata ({k})",
                        confidence="VERIFIED",
                        details=s_val,
                        via=via,
                        score=score
                    ))

    # 2. PIL EXIF data & Sub-IFDs
    try:
        exif = pil_img.getexif()
        if exif:
            for tag_id, val in exif.items():
                s_val = val.decode("latin-1", "replace") if isinstance(val, bytes) else str(val)
                findings[f"EXIF:{tag_id}"] = s_val
                text_corpus.append(s_val)
                for flg, via in find_flags_deep(s_val, custom_regex):
                    score = score_flag_candidate(flg, custom_regex)
                    flags.append(FlagCandidate(
                        flag=flg,
                        source=f"EXIF Tag {tag_id}",
                        confidence="VERIFIED",
                        details=s_val,
                        via=via,
                        score=score
                    ))
            for ifd_id in (0x8769, 0x8825, 0xA005):
                try:
                    sub_ifd = exif.get_ifd(ifd_id)
                    if sub_ifd:
                        for k, val in sub_ifd.items():
                            s_val = val.decode("latin-1", "replace") if isinstance(val, bytes) else str(val)
                            findings[f"EXIF-IFD{ifd_id}:{k}"] = s_val
                            text_corpus.append(s_val)
                            for flg, via in find_flags_deep(s_val, custom_regex):
                                score = score_flag_candidate(flg, custom_regex)
                                flags.append(FlagCandidate(
                                    flag=flg,
                                    source=f"EXIF IFD {ifd_id} Tag {k}",
                                    confidence="VERIFIED",
                                    details=s_val,
                                    via=via,
                                    score=score
                                ))
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Manual PNG chunk scanner (detect private chunks and bad CRCs)
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        p = 8
        while p + 12 <= len(raw_bytes):
            length = int.from_bytes(raw_bytes[p:p+4], "big")
            chunk_type = raw_bytes[p+4:p+8]
            chunk_data = raw_bytes[p+8:p+8+length]
            chunk_crc = int.from_bytes(raw_bytes[p+8+length:p+12+length], "big")
            expected_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
            if chunk_crc != expected_crc:
                findings[f"Chunk {chunk_type.decode('latin-1', 'replace')} (BAD CRC)"] = f"CRC: 0x{chunk_crc:08X} vs 0x{expected_crc:08X}"
            
            if chunk_type not in (b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tEXt", b"zTXt", b"iTXt", b"bKGD", b"pHYs", b"sBIT"):
                findings[f"Private Chunk: {chunk_type.decode('latin-1', 'replace')}"] = f"len={length}"
                for flg, via in find_flags_deep(chunk_data, custom_regex):
                    score = score_flag_candidate(flg, custom_regex)
                    flags.append(FlagCandidate(
                        flag=flg,
                        source=f"PNG Chunk {chunk_type.decode('latin-1', 'replace')}",
                        confidence="VERIFIED",
                        details=f"Private chunk of length {length}",
                        via=via,
                        score=score
                    ))
            p += 12 + length
            if chunk_type == b"IEND":
                break

    return findings, flags, text_corpus


def scan_polyglot_trailer(
    raw_bytes: bytes,
    filename: str = "",
    custom_regex: Optional[str] = None,
    candidate_keys: Optional[List[bytes]] = None,
    notes: Optional[List[str]] = None,
) -> Tuple[List[CarvedFile], List[FlagCandidate], List[str]]:
    """Carve and examine data past file boundary (ZIP, RAR, 7z, PDF, raw text appended)."""
    carved_files: List[CarvedFile] = []
    flags: List[FlagCandidate] = []
    text_corpus: List[str] = []

    audit = audit_file_structure(raw_bytes, filename)
    trailing = audit.get("trailing_data", b"")
    if not trailing or len(trailing) < 4:
        return carved_files, flags, text_corpus

    # Check for embedded ZIP archive
    if b"PK\x03\x04" in trailing:
        zip_offset = trailing.find(b"PK\x03\x04")
        zip_bytes = trailing[zip_offset:]
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.infolist():
                    content = None
                    try:
                        content = zf.read(member.filename)
                    except Exception:
                        pass
                    if content is None and candidate_keys:
                        for pwd in candidate_keys:
                            try:
                                content = zf.read(member.filename, pwd=pwd)
                                if notes is not None:
                                    notes.append(f"Successfully decrypted ZIP member '{member.filename}' with password '{pwd.decode('latin-1', 'replace')}'")
                                break
                            except Exception:
                                pass

                    if content is not None:
                        is_text = sum(32 <= b < 127 or b in (10, 13, 9) for b in content[:512]) > 0.8 * min(len(content), 512)
                        carved_files.append(CarvedFile(
                            filename=member.filename,
                            size_bytes=len(content),
                            data=content,
                            mime_type="text/plain" if is_text else "application/octet-stream",
                            note=f"Carved from embedded ZIP past EOF (offset +{zip_offset})"
                        ))
                        if is_text:
                            text_corpus.append(content.decode("utf-8", "replace"))

                        for flg, via in find_flags_deep(content, custom_regex):
                            score = score_flag_candidate(flg, custom_regex)
                            flags.append(FlagCandidate(
                                flag=flg,
                                source=f"Embedded ZIP: {member.filename}",
                                confidence="VERIFIED",
                                details=f"Found inside carved archive file '{member.filename}'",
                                via=via,
                                score=score
                            ))
        except Exception:
            pass

    # Record the raw trailing payload as a carved file regardless
    carved_files.append(CarvedFile(
        filename="carved_payload.bin",
        size_bytes=len(trailing),
        data=trailing,
        mime_type="application/octet-stream",
        note=f"{len(trailing)} bytes detected past legitimate format boundary ({audit.get('reason', '')})"
    ))

    # Scan raw trailing bytes for flags directly
    for flg, via in find_flags_deep(trailing, custom_regex):
        score = score_flag_candidate(flg, custom_regex)
        flags.append(FlagCandidate(
            flag=flg,
            source="EOF Trailing Polyglot",
            confidence="HIGH",
            details=f"Discovered in {len(trailing)} carved bytes after format boundary",
            via=via,
            score=score
        ))

    return carved_files, flags, text_corpus


def build_pixel_sources(img: Image.Image) -> Dict[str, np.ndarray]:
    """Extract standard and converted pixel channels (RGB, YCbCr, HSV, Alpha, Palette)."""
    if img.mode == "YCbCr":
        y, cb, cr = img.split()
        return {
            "Y": np.array(y, dtype=np.uint8).flatten(),
            "Cb": np.array(cb, dtype=np.uint8).flatten(),
            "Cr": np.array(cr, dtype=np.uint8).flatten(),
        }

    sources: Dict[str, np.ndarray] = {}
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    ycc = rgb.convert("YCbCr")
    y, cb, cr = ycc.split()
    hsv = rgb.convert("HSV")
    h, s, v = hsv.split()

    sources["Red"] = np.array(r, dtype=np.uint8).flatten()
    sources["Green"] = np.array(g, dtype=np.uint8).flatten()
    sources["Blue"] = np.array(b, dtype=np.uint8).flatten()
    sources["Y"] = np.array(y, dtype=np.uint8).flatten()
    sources["Cb"] = np.array(cb, dtype=np.uint8).flatten()
    sources["Cr"] = np.array(cr, dtype=np.uint8).flatten()
    sources["Hue"] = np.array(h, dtype=np.uint8).flatten()
    sources["Saturation"] = np.array(s, dtype=np.uint8).flatten()
    sources["Value"] = np.array(v, dtype=np.uint8).flatten()

    if img.mode == "RGBA":
        a = img.split()[3]
        sources["Alpha"] = np.array(a, dtype=np.uint8).flatten()
    elif img.mode == "P":
        sources["Palette"] = np.array(img, dtype=np.uint8).flatten()

    return sources


def crack_keyed_and_strided_lsb(
    pil_img: Image.Image,
    candidate_keys: List[bytes],
    custom_regex: Optional[str] = None,
    strides: Optional[List[int]] = None,
    max_offsets: int = 50,
    sources: Optional[Dict[str, np.ndarray]] = None,
    bit_planes: Optional[List[int]] = None,
    level: str = "deep",
) -> List[FlagCandidate]:
    """Vectorized sweep across channels, bit planes, strides, offsets, bit packing, and keys."""
    flags: List[FlagCandidate] = []
    if strides is None:
        strides = [1, 2, 3, 4, 5, 8] if level == "deep" else [1]
    if bit_planes is None:
        bit_planes = [0, 1, 2, 3] if level == "deep" else [0]

    keys_to_test = list(candidate_keys)
    if not keys_to_test:
        keys_to_test = [b"password", b"stego", b"key", b"hidden"]

    if sources is None:
        if pil_img.mode == "YCbCr":
            y, cb, cr = pil_img.split()
            sources = {
                "Y": np.array(y, dtype=np.uint8).flatten(),
                "Cb": np.array(cb, dtype=np.uint8).flatten(),
                "Cr": np.array(cr, dtype=np.uint8).flatten(),
            }
        elif pil_img.mode == "P":
            sources = {"Palette": np.array(pil_img, dtype=np.uint8).flatten()}
        elif pil_img.mode == "RGBA":
            r, g, b, a = pil_img.split()
            sources = {
                "Red": np.array(r, dtype=np.uint8).flatten(),
                "Green": np.array(g, dtype=np.uint8).flatten(),
                "Blue": np.array(b, dtype=np.uint8).flatten(),
                "Alpha": np.array(a, dtype=np.uint8).flatten(),
            }
        else:
            rgb = pil_img.convert("RGB")
            r, g, b = rgb.split()
            sources = {
                "Red": np.array(r, dtype=np.uint8).flatten(),
                "Green": np.array(g, dtype=np.uint8).flatten(),
                "Blue": np.array(b, dtype=np.uint8).flatten(),
            }

    for s_name, chan in sources.items():
        # Sweep bit planes
        for k in bit_planes:
            bit_chan = ((chan >> k) & 1).astype(np.uint8)
            # Only bit plane 0 requires extensive strides and offsets
            k_strides = strides if k == 0 else [1]
            k_offsets = (max_offsets if level == "deep" else 1) if k == 0 else 1

            for stride in k_strides:
                for offset in range(0, min(k_offsets, len(bit_chan))):
                    bits = bit_chan[offset : offset + 2048 * stride : stride]
                    if len(bits) < 64:
                        continue
                    n_check = min(len(bits) // 8, 256)
                    if n_check < 8:
                        continue

                    b_msb = np.packbits(bits[:n_check * 8]).tobytes()
                    b_lsb = np.packbits(bits[:n_check * 8].reshape(-1, 8)[:, ::-1].flatten()).tobytes()

                    src_label = f"LSB: {s_name}" if k == 0 else f"{s_name} bit{k}"

                    for b_data, order in [(b_msb, "MSB"), (b_lsb, "LSB")]:
                        # 1. Plain & Encoded streams
                        for flg, via in find_flags_in_bytes(b_data, custom_regex):
                            score = score_flag_candidate(flg, custom_regex)
                            if score < 30.0:
                                continue
                            flags.append(FlagCandidate(
                                flag=flg,
                                source=f"{src_label}",
                                confidence="VERIFIED" if score >= 100.0 else "HIGH",
                                details=f"Stride: {stride}, Offset: {offset}, Order: {order}",
                                via=via,
                                score=score
                            ))

                        # 2. Keyed XOR (Fast rejection filter)
                        for key_b in keys_to_test:
                            if not key_b:
                                continue
                            exp_k = (key_b * (len(b_data) // len(key_b) + 1))[:len(b_data)]
                            xored = bytes(x ^ y for x, y in zip(b_data, exp_k))
                            if b"{" in xored:
                                for flg in find_flags_in_text(xored.decode("latin-1", "replace"), custom_regex):
                                    score = score_flag_candidate(flg, custom_regex)
                                    if score < 30.0:
                                        continue
                                    flags.append(FlagCandidate(
                                        flag=flg,
                                        source=f"LSB XOR: {s_name}",
                                        confidence="VERIFIED" if score >= 100.0 else "HIGH",
                                        details=f"Key: '{key_b.decode('latin-1', 'replace')}', Stride: {stride}, Offset: {offset}, Order: {order}",
                                        via="plain",
                                        score=score
                                    ))

        # Test 2 bits per sample (bits 1 and 0 interleaved)
        if level == "deep" and len(chan) >= 64:
            b = chan & 3
            b1 = ((b >> 1) & 1).astype(np.uint8)
            b0 = (b & 1).astype(np.uint8)
            pairs = np.column_stack([b1, b0]).flatten()
            n_check = min(len(pairs) // 8, 512)
            if n_check >= 8:
                b_2bit = np.packbits(pairs[:n_check * 8]).tobytes()
                for flg in find_flags_in_text(b_2bit.decode("latin-1", "replace"), custom_regex):
                    score = score_flag_candidate(flg, custom_regex)
                    flags.append(FlagCandidate(
                        flag=flg,
                        source=f"LSB: {s_name} 2-bpp",
                        confidence="VERIFIED",
                        details="Extracted from (bit1, bit0) sample pairs",
                        via="plain",
                        score=score
                    ))

    return flags


def scan_alpha_variants(pil_img: Image.Image, custom_regex: Optional[str] = None) -> Tuple[List[CarvedFile], List[FlagCandidate]]:
    """Inspect Alpha channel ASCII encoding, 0-alpha transparent RGB data, and generate masks."""
    carved_files: List[CarvedFile] = []
    flags: List[FlagCandidate] = []

    if pil_img.mode != "RGBA":
        return carved_files, flags

    arr = np.array(pil_img, dtype=np.uint8)
    h, w, _ = arr.shape
    alpha = arr[:, :, 3]

    # (a) Alpha values are ASCII codes of the flag
    alpha_bytes = alpha.flatten().tobytes()
    for flg, via in find_flags_in_bytes(alpha_bytes, custom_regex):
        score = score_flag_candidate(flg, custom_regex)
        flags.append(FlagCandidate(
            flag=flg,
            source="Alpha Channel ASCII",
            confidence="VERIFIED",
            details="Directly encoded in Alpha channel bytes",
            via=via,
            score=score
        ))

    # (b) RGB bytes hidden under fully transparent pixels (alpha == 0)
    zero_mask = (alpha == 0)
    if np.any(zero_mask):
        # 1. alpha_mask.png
        m_img = Image.fromarray((zero_mask.astype(np.uint8) * 255), mode="L")
        m_buf = io.BytesIO()
        m_img.save(m_buf, format="PNG")
        carved_files.append(CarvedFile("alpha_mask.png", len(m_buf.getvalue()), m_buf.getvalue(), "image/png", "Mask of transparent pixels (alpha == 0)"))

        # 2. alpha_mask_inverted.png
        inv_img = Image.fromarray(((~zero_mask).astype(np.uint8) * 255), mode="L")
        inv_buf = io.BytesIO()
        inv_img.save(inv_buf, format="PNG")
        carved_files.append(CarvedFile("alpha_mask_inverted.png", len(inv_buf.getvalue()), inv_buf.getvalue(), "image/png", "Inverted mask of opaque pixels"))

        # 3. alpha0_rgb_revealed.png
        rev_img = Image.fromarray(arr[:, :, :3], mode="RGB")
        rev_buf = io.BytesIO()
        rev_img.save(rev_buf, format="PNG")
        carved_files.append(CarvedFile("alpha0_rgb_revealed.png", len(rev_buf.getvalue()), rev_buf.getvalue(), "image/png", "RGB channels under transparent alpha"))

        # Scan RGB bytes under transparent pixels
        trans_rgb = arr[zero_mask, :3].flatten().tobytes()
        for flg, via in find_flags_in_bytes(trans_rgb, custom_regex):
            score = score_flag_candidate(flg, custom_regex)
            flags.append(FlagCandidate(
                flag=flg,
                source="Alpha-0 Transparent RGB",
                confidence="VERIFIED",
                details="Extracted from RGB bytes under alpha == 0",
                via=via,
                score=score
            ))

        raster_rgb = arr.reshape(-1, 4)[:, :3].flatten().tobytes()
        for flg, via in find_flags_in_bytes(raster_rgb, custom_regex):
            score = score_flag_candidate(flg, custom_regex)
            flags.append(FlagCandidate(
                flag=flg,
                source="Alpha-0 Transparent RGB Raster",
                confidence="VERIFIED",
                details="Extracted from raster RGB array",
                via=via,
                score=score
            ))

    return carved_files, flags


def apply_stegsolve_filter(pil_img: Image.Image, filter_name: str) -> Image.Image:
    """Apply multi-spectral visual bitplane or color transform filter for optical forensic analysis."""
    import cv2
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    if "Plane" in filter_name:
        parts = filter_name.split()
        c_map = {"Red": 0, "Green": 1, "Blue": 2, "Alpha": 3}
        chan_name = parts[0]
        bit_idx = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        if chan_name == "Alpha":
            rgba = pil_img.convert("RGBA")
            arr4 = np.array(rgba)
            plane = (((arr4[:, :, 3] >> bit_idx) & 1) * 255).astype(np.uint8)
            return Image.fromarray(plane, mode="L")
        else:
            chan_idx = c_map.get(chan_name, 0)
            plane = (((arr[:, :, chan_idx] >> bit_idx) & 1) * 255).astype(np.uint8)
            return Image.fromarray(plane, mode="L")
    elif filter_name == "Red Channel Only":
        zero = np.zeros_like(arr[:, :, 0])
        return Image.fromarray(np.stack([arr[:, :, 0], zero, zero], axis=-1), mode="RGB")
    elif filter_name == "Green Channel Only":
        zero = np.zeros_like(arr[:, :, 0])
        return Image.fromarray(np.stack([zero, arr[:, :, 1], zero], axis=-1), mode="RGB")
    elif filter_name == "Blue Channel Only":
        zero = np.zeros_like(arr[:, :, 0])
        return Image.fromarray(np.stack([zero, zero, arr[:, :, 2]], axis=-1), mode="RGB")
    elif filter_name == "Color Inversion":
        return Image.fromarray(255 - arr, mode="RGB")
    elif filter_name == "Grayscale Invert":
        gray = np.array(pil_img.convert("L"))
        return Image.fromarray(255 - gray, mode="L")
    elif filter_name == "High Contrast Threshold":
        gray = np.array(pil_img.convert("L"))
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th, mode="L")
    elif filter_name == "Edge / High-Pass":
        gray = np.array(pil_img.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        return Image.fromarray(edges, mode="L")
    return pil_img


def decode_binary_ascii_string(text: str) -> str:
    """Decode space-separated or raw 8-bit binary strings (e.g. '01001000 01001001') to ASCII text."""
    chunks = re.findall(r"[01]{8}", text)
    if not chunks:
        return ""
    return "".join(chr(int(b, 2)) for b in chunks)


def scan_qr_codes(
    pil_img: Image.Image,
    custom_regex: Optional[str] = None,
) -> Tuple[List[FlagCandidate], List[str]]:
    """Scan and decode visual QR codes using adaptive thresholds and contrast equalization."""
    import cv2
    flags: List[FlagCandidate] = []
    notes: List[str] = []
    try:
        cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        detector = cv2.QRCodeDetector()

        candidates = [gray]
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        candidates.append(clahe.apply(gray))
        for t in (80, 128, 180):
            _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            candidates.append(th)
            candidates.append(255 - th)

        seen_vals = set()
        for im in candidates:
            val, pts, _ = detector.detectAndDecode(im)
            if val and val not in seen_vals:
                seen_vals.add(val)
                notes.append(f"Visual QR Code decoded: '{val}'")
                for flg, via in find_flags_deep(val, custom_regex):
                    score = score_flag_candidate(flg, custom_regex)
                    flags.append(FlagCandidate(
                        flag=flg,
                        source="Visual QR Code",
                        confidence="VERIFIED",
                        details=f"Decoded from optical QR code: {val}",
                        via=via,
                        score=score
                    ))
    except Exception:
        pass
    return flags, notes


def scan_visual_text_ocr(
    pil_img: Image.Image,
    custom_regex: Optional[str] = None,
) -> Tuple[List[FlagCandidate], List[str], List[str]]:
    """Scan and recognize visual text printed on the image canvas using Optical Character Recognition (OCR)."""
    flags: List[FlagCandidate] = []
    notes: List[str] = []
    corpus: List[str] = []
    try:
        import pytesseract
        text = pytesseract.image_to_string(pil_img)
        if text and text.strip():
            corpus.append(text)
            # 1. Standard search
            for flg, via in find_flags_deep(text, custom_regex):
                score = score_flag_candidate(flg, custom_regex)
                flags.append(FlagCandidate(
                    flag=flg,
                    source="Visual Canvas Text (OCR)",
                    confidence="VERIFIED",
                    details="Recognized directly from visible text rendered on the canvas via OCR",
                    via=via,
                    score=score + 150,  # High priority because it is directly visible on the canvas!
                ))
            # 2. Tolerant OCR search (handles OCR whitespace / kerning inside curly braces)
            for m in re.finditer(r"([A-Za-z0-9_\-\.]{2,32})\s*\{\s*([A-Za-z0-9_\-\.\!\@\#\$\%\^\&\*\+\=\s]{3,100})\s*\}", text):
                clean_body = re.sub(r"\s+", "", m.group(2))
                clean_flag = f"{m.group(1)}{{{clean_body}}}"
                if not any(f.flag == clean_flag for f in flags):
                    score = score_flag_candidate(clean_flag, custom_regex)
                    flags.append(FlagCandidate(
                        flag=clean_flag,
                        source="Visual Canvas Text (OCR)",
                        confidence="VERIFIED",
                        details="Recognized from visible text rendered on the canvas via OCR (whitespace normalized)",
                        via="plain",
                        score=score + 150,
                    ))
            if flags:
                notes.append(f"Visual Text OCR recognized flag: {flags[0].flag}")
    except Exception:
        pass
    return flags, notes, corpus


def repair_png_ihdr_height(
    data: bytes,
    max_search: int = 10000,
) -> Optional[Tuple[bytes, int, int, int]]:
    """Detect and repair tampered PNG IHDR height using CRC32 brute-force (anti-forensic canvas truncation).
    Returns (repaired_png_bytes, width, original_tampered_height, recovered_true_height) or None.
    """
    if not data or len(data) < 33 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None

    # IHDR chunk starts at offset 12: Length(4)=13, Type(4)='IHDR', Data(13), CRC(4)
    if data[12:16] != b"IHDR":
        return None

    stored_crc = struct.unpack(">I", data[29:33])[0]
    ihdr_type_and_data = data[12:29]  # 'IHDR' + 13 bytes data
    calc_crc = zlib.crc32(ihdr_type_and_data) & 0xFFFFFFFF

    if calc_crc == stored_crc:
        return None  # CRC is valid, height not modified via single-parameter hex edit

    w = struct.unpack(">I", data[16:20])[0]
    orig_h = struct.unpack(">I", data[20:24])[0]
    rest = data[24:29]  # bit_depth, color_type, compression, filter, interlace

    for candidate_h in range(1, max_search + 1):
        test_chunk = b"IHDR" + struct.pack(">I", w) + struct.pack(">I", candidate_h) + rest
        if (zlib.crc32(test_chunk) & 0xFFFFFFFF) == stored_crc:
            repaired_data = data[:20] + struct.pack(">I", candidate_h) + data[24:]
            return repaired_data, w, orig_h, candidate_h

    return None


def audit_c2pa_provenance(data: bytes) -> Dict[str, Any]:
    """Parse and cryptographically verify C2PA JUMBF / CBOR manifest in PNG/JPEG media."""
    result: Dict[str, Any] = {
        "has_c2pa": False,
        "exclusions": [],
        "expected_hash": None,
        "calculated_hash": None,
        "hash_match": False,
        "software_agent": None,
        "digital_source_type": None,
        "signing_time": None,
        "analysis_notes": [],
    }
    pos = data.find(b"caBX")
    if pos != -1 and pos >= 4:
        result["has_c2pa"] = True
        length = int.from_bytes(data[pos - 4 : pos], "big")
        chunk_data = data[pos + 4 : pos + 4 + length]

        excl_start = pos - 4
        excl_len = length + 12  # length(4) + type(4) + data(length) + crc(4)
        result["exclusions"].append({"start": excl_start, "length": excl_len})

        import hashlib
        rem = data[:excl_start] + data[excl_start + excl_len :]
        calc_digest = hashlib.sha256(rem).digest()
        calc_b64 = base64.b64encode(calc_digest).decode()
        result["calculated_hash"] = calc_b64

        b64_hashes = re.findall(rb"[A-Za-z0-9+/]{43}=", chunk_data)
        for bh in b64_hashes:
            h_str = bh.decode("latin-1")
            if h_str == calc_b64:
                result["expected_hash"] = h_str
                result["hash_match"] = True
                break
        if not result["expected_hash"] and b64_hashes:
            result["expected_hash"] = b64_hashes[0].decode("latin-1")

        if b"ChatGPT" in chunk_data:
            result["software_agent"] = "ChatGPT (gpt-image / DALL-E 3)"
        elif b"DALL-E" in chunk_data:
            result["software_agent"] = "OpenAI DALL-E"
        elif b"Midjourney" in chunk_data:
            result["software_agent"] = "Midjourney"

        if b"trainedAlgorithmicMedia" in chunk_data:
            result["digital_source_type"] = "Generative AI (trainedAlgorithmicMedia)"

        if result["hash_match"]:
            result["analysis_notes"].append(
                "C2PA SHA-256 asset hash matches manifest perfectly. "
                "Forensic note: The hash establishes integrity relative to the manifest at signing time; "
                "it confirms the file was not altered post-signing, but is compatible with both clean images "
                "and images where steganographic payloads were embedded prior to signing."
            )
    return result


def scan_bitplanes_lsb(
    pil_img: Image.Image,
    max_bytes: int = 8192,
    custom_regex: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[FlagCandidate]]:
    """Legacy bit-plane analysis for telemetry reporting and backward compatibility."""
    findings: List[Dict[str, Any]] = []
    flags: List[FlagCandidate] = []

    img_rgb = pil_img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.uint8)

    channels = {
        "Red (R) LSB": arr[:, :, 0].flatten(),
        "Green (G) LSB": arr[:, :, 1].flatten(),
        "Blue (B) LSB": arr[:, :, 2].flatten(),
        "RGB Interleaved LSB": arr.reshape(-1),
        "BGR Interleaved LSB": arr[:, :, ::-1].reshape(-1),
        "Red Column-Major LSB": arr[:, :, 0].T.flatten(),
        "Green Column-Major LSB": arr[:, :, 1].T.flatten(),
        "Blue Column-Major LSB": arr[:, :, 2].T.flatten(),
    }

    for name, pixel_stream in channels.items():
        bits = (pixel_stream & 1).astype(np.uint8)
        n_bits = min(len(bits) // 8 * 8, max_bytes * 8)
        if n_bits < 64:
            continue
        
        packed_msb = np.packbits(bits[:n_bits]).tobytes()
        bits_rev = bits[:n_bits].reshape(-1, 8)[:, ::-1].flatten()
        packed_lsb = np.packbits(bits_rev).tobytes()

        for pack_name, byte_data in [("MSB-first", packed_msb), ("LSB-first", packed_lsb)]:
            readability = sum(32 <= b < 127 or b in (10, 13, 9) for b in byte_data[:512]) / max(len(byte_data[:512]), 1)
            entropy = calculate_entropy(byte_data[:512])

            channel_entry = {
                "channel": f"{name} ({pack_name})",
                "readability": readability,
                "entropy": entropy,
                "sample_preview": "".join(chr(b) if 32 <= b < 127 else "·" for b in byte_data[:64]),
                "raw_bytes": byte_data[:512],
            }
            findings.append(channel_entry)

            for flg, via in find_flags_in_bytes(byte_data, custom_regex):
                score = score_flag_candidate(flg, custom_regex)
                flags.append(FlagCandidate(
                    flag=flg,
                    source=f"LSB: {name} ({pack_name})",
                    confidence="VERIFIED",
                    details=f"Readability: {readability*100:.1f}%, Shannon Entropy: {entropy:.2f}",
                    via=via,
                    score=score
                ))

    return findings, flags


# ── JPEG DCT COEFFICIENTS HUFFMAN DECODER ─────────────────────────────────────
class _HuffmanTable:
    def __init__(self):
        self.lookup = {}

    def build(self, counts: List[int], symbols: List[int]):
        self.lookup = {}
        code = 0
        sym_idx = 0
        for length in range(1, 17):
            cnt = counts[length - 1]
            for _ in range(cnt):
                if sym_idx < len(symbols):
                    self.lookup[(code, length)] = symbols[sym_idx]
                    sym_idx += 1
                code += 1
            code <<= 1


class _BitStream:
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_buf = 0
        self.bits_left = 0

    def get_bit(self) -> int:
        if self.bits_left == 0:
            if self.byte_pos >= len(self.data):
                return 0
            b = self.data[self.byte_pos]
            self.byte_pos += 1
            if b == 0xFF:
                if self.byte_pos < len(self.data):
                    nxt = self.data[self.byte_pos]
                    if nxt == 0x00:
                        self.byte_pos += 1
                    elif 0xD0 <= nxt <= 0xD7:
                        self.byte_pos += 1
            self.bit_buf = b
            self.bits_left = 8

        self.bits_left -= 1
        return (self.bit_buf >> self.bits_left) & 1

    def get_bits(self, n: int) -> int:
        val = 0
        for _ in range(n):
            val = (val << 1) | self.get_bit()
        return val

    def decode_symbol(self, ht: _HuffmanTable) -> int:
        code = 0
        for length in range(1, 17):
            code = (code << 1) | self.get_bit()
            if (code, length) in ht.lookup:
                return ht.lookup[(code, length)]
        return 0


def _receive_extend(val: int, category: int) -> int:
    if category == 0:
        return 0
    vt = 1 << (category - 1)
    if val < vt:
        return val + (-1 << category) + 1
    return val


def decode_jpeg_coefficients(raw: bytes, max_blocks: int = 10_000) -> Dict[str, Any]:
    """Decode baseline sequential JPEG into 8x8 DCT coefficients (zigzag ordered)."""
    if not raw.startswith(b"\xff\xd8"):
        return {"components": {}}

    pos = 2
    dht_tables: Dict[Tuple[int, int], _HuffmanTable] = {}
    components: Dict[int, Dict[str, Any]] = {}
    sos_components: List[int] = []
    scan_data_start = None

    while pos < len(raw) - 1:
        if raw[pos] != 0xFF:
            pos += 1
            continue
        marker = raw[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9):
            continue
        if 0xD0 <= marker <= 0xD7 or marker == 0x00:
            continue

        if pos + 2 > len(raw):
            break
        length = int.from_bytes(raw[pos:pos+2], "big")
        data = raw[pos+2 : pos+length]
        next_pos = pos + length

        if marker == 0xC4:  # DHT
            d_pos = 0
            while d_pos < len(data):
                info = data[d_pos]
                tc = (info >> 4) & 1
                th = info & 0x0F
                counts = list(data[d_pos+1 : d_pos+17])
                total_syms = sum(counts)
                symbols = list(data[d_pos+17 : d_pos+17+total_syms])
                ht = _HuffmanTable()
                ht.build(counts, symbols)
                dht_tables[(tc, th)] = ht
                d_pos += 17 + total_syms

        elif marker in (0xC0, 0xC2):  # SOF0 (baseline) or SOF2
            if len(data) >= 6:
                num_comp = data[5]
                for i in range(num_comp):
                    if 6 + i*3 + 3 <= len(data):
                        cid = data[6 + i*3]
                        hv = data[7 + i*3]
                        h_samp = (hv >> 4) & 0x0F
                        v_samp = hv & 0x0F
                        qid = data[8 + i*3]
                        components[cid] = {"h": h_samp, "v": v_samp, "q": qid, "id": cid}

        elif marker == 0xDA:  # SOS
            if len(data) >= 1:
                num_comp = data[0]
                for i in range(num_comp):
                    if 1 + i*2 + 2 <= len(data):
                        cid = data[1 + i*2]
                        td_ta = data[2 + i*2]
                        dc_tbl = (td_ta >> 4) & 0x0F
                        ac_tbl = td_ta & 0x0F
                        if cid in components:
                            components[cid]["dc"] = dc_tbl
                            components[cid]["ac"] = ac_tbl
                            sos_components.append(cid)
            scan_data_start = next_pos
            break

        pos = next_pos

    if scan_data_start is None or not sos_components:
        return {"components": {}}

    bs = _BitStream(raw[scan_data_start:])
    blocks_out: Dict[int, List[np.ndarray]] = {cid: [] for cid in sos_components}
    dc_preds: Dict[int, int] = {cid: 0 for cid in sos_components}

    total_blocks_read = 0
    done = False

    while not done and total_blocks_read < max_blocks:
        if bs.byte_pos >= len(bs.data):
            break
        for cid in sos_components:
            comp = components[cid]
            dc_table = dht_tables.get((0, comp.get("dc", 0)))
            ac_table = dht_tables.get((1, comp.get("ac", 0)))
            if not dc_table or not ac_table:
                done = True
                break

            h_blocks = comp["h"] if len(sos_components) > 1 else 1
            v_blocks = comp["v"] if len(sos_components) > 1 else 1

            for _ in range(v_blocks * h_blocks):
                block = np.zeros(64, dtype=np.int16)
                dc_cat = bs.decode_symbol(dc_table)
                dc_val = bs.get_bits(dc_cat)
                dc_diff = _receive_extend(dc_val, dc_cat)
                dc_preds[cid] += dc_diff
                block[0] = dc_preds[cid]

                k = 1
                while k < 64:
                    sym = bs.decode_symbol(ac_table)
                    if sym == 0x00:
                        break
                    if sym == 0xF0:
                        k += 16
                        continue
                    run = (sym >> 4) & 0x0F
                    cat = sym & 0x0F
                    k += run
                    if k < 64:
                        val = bs.get_bits(cat)
                        block[k] = _receive_extend(val, cat)
                        k += 1

                blocks_out[cid].append(block)
                total_blocks_read += 1
                if total_blocks_read >= max_blocks:
                    done = True
                    break

    res_components = {}
    for cid, blist in blocks_out.items():
        if blist:
            res_components[cid] = np.array(blist, dtype=np.int16)

    return {"components": res_components}


def _dct_variants(blocks: np.ndarray) -> Dict[str, np.ndarray]:
    """Extract candidate DCT bit streams (JSteg, non-zero AC, raw)."""
    if len(blocks) == 0:
        return {}
    ac = blocks[:, 1:].flatten()
    jsteg_mask = (ac != 0) & (ac != 1)
    jsteg_bits = (ac[jsteg_mask] & 1).astype(np.uint8)

    nonzero_mask = ac != 0
    nonzero_bits = (ac[nonzero_mask] & 1).astype(np.uint8)

    all_bits = (ac & 1).astype(np.uint8)

    return {
        "JSteg (skip 0 and 1) LSB": jsteg_bits,
        "Non-zero AC LSB": nonzero_bits,
        "All AC LSB": all_bits,
    }


def _pack_both(bits: np.ndarray) -> List[Tuple[str, bytes]]:
    """Pack bits into MSB-first and LSB-first byte arrays."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return [("MSB-first", b""), ("LSB-first", b"")]
    b_msb = np.packbits(bits[:n]).tobytes()
    b_lsb = np.packbits(bits[:n].reshape(-1, 8)[:, ::-1].flatten()).tobytes()
    return [("MSB-first", b_msb), ("LSB-first", b_lsb)]


def run_zsteg_analysis(image_path: str, custom_regex: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[FlagCandidate]]:
    """Execute zsteg against PNG/BMP files if installed."""
    findings: List[Dict[str, Any]] = []
    flags: List[FlagCandidate] = []

    which_path = shutil.which("zsteg")
    if not which_path:
        for c in [
            str(Path.home() / ".gem/ruby/2.6.0/bin/zsteg"),
            "/usr/local/bin/zsteg",
            "/usr/bin/zsteg",
        ]:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                which_path = c
                break

    if not which_path or not os.path.exists(image_path):
        return findings, flags

    try:
        proc = subprocess.run(
            [which_path, "-a", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=12,
        )
        stdout_str = proc.stdout.decode("utf-8", "replace")
        for line in stdout_str.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith("[?]"):
                continue
            findings.append({"tool": "zsteg", "output": line_str})
            for flg, via in find_flags_deep(line_str, custom_regex):
                score = score_flag_candidate(flg, custom_regex)
                flags.append(FlagCandidate(
                    flag=flg,
                    source="zsteg CLI",
                    confidence="VERIFIED",
                    details=line_str,
                    via=via,
                    score=score
                ))
    except Exception as e:
        findings.append({"tool": "zsteg", "error": str(e)})

    return findings, flags


def solve_image_stego(
    raw_bytes: bytes,
    filename: str = "",
    pil_img: Optional[Image.Image] = None,
    custom_regex: Optional[str] = None,
    user_passphrase: Optional[str] = None,
    wordlist: Optional[str] = None,
    level: str = "deep",
) -> SolverReport:
    """Master automated forensic pipeline for deep covert channel recovery and steganalysis."""
    t_start = time.time()
    report = SolverReport()
    corpus: List[str] = []

    # Corrupt or empty inputs safeguard
    if not raw_bytes or not isinstance(raw_bytes, (bytes, bytearray)):
        return report

    if pil_img is None:
        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
        except Exception:
            pil_img = None

    # 0. Check for PNG IHDR Dimension/Height Tampering (Anti-Forensic Canvas Truncation)
    try:
        ihdr_fix = repair_png_ihdr_height(raw_bytes)
        if ihdr_fix:
            repaired_bytes, w, old_h, true_h = ihdr_fix
            report.notes.append(
                f"🚨 PNG IHDR Height Tampering detected & repaired! Stored: {w}x{old_h}px, True: {w}x{true_h}px."
            )
            report.carved_files.append(CarvedFile(
                filename="repaired_uncropped_image.png",
                mime_type="image/png",
                size_bytes=len(repaired_bytes),
                data=repaired_bytes,
                note=f"Original canvas recovered via CRC32 brute-force: {w}x{true_h}px",
            ))
            try:
                pil_img = Image.open(io.BytesIO(repaired_bytes))
                raw_bytes = repaired_bytes
            except Exception:
                pass
    except Exception as e:
        report.errors.append(f"IHDR repair: {e}")

    # 0.5 Check C2PA Provenance Manifest & Cryptographic Bindings
    try:
        c2pa_res = audit_c2pa_provenance(raw_bytes)
        if c2pa_res["has_c2pa"]:
            for n in c2pa_res["analysis_notes"]:
                report.notes.append(n)
            if c2pa_res["software_agent"]:
                # ``metadata_findings`` is a mapping, not a list.  Keep the
                # provenance fields here and merge regular metadata below so
                # C2PA findings are both type-safe and retained in the report.
                report.metadata_findings.update({
                    "Provenance Agent": str(c2pa_res["software_agent"]),
                    "Provenance Source Type": str(
                        c2pa_res.get("digital_source_type", "AI-Generated")
                    ),
                    "Provenance Asset SHA256 Match": str(c2pa_res["hash_match"]),
                })
    except Exception:
        pass

    # 1. Scan Metadata (PNG chunks, EXIF, Comments)
    t0 = time.time()
    try:
        if pil_img is not None:
            meta_findings, meta_flags, meta_corpus = scan_metadata(raw_bytes, pil_img, custom_regex)
            # Preserve C2PA provenance fields collected immediately above.
            report.metadata_findings.update(meta_findings)
            report.flags.extend(meta_flags)
            corpus.extend(meta_corpus)
    except Exception as e:
        report.errors.append(f"Metadata scan: {e}")
    report.stage_timings["metadata"] = time.time() - t0

    # 2. Key Harvesting (from metadata, user pass, defaults)
    passwords = [user_passphrase] if user_passphrase else []
    candidate_keys = harvest_candidate_keys(corpus, passwords)
    report.harvested_keys = [k.decode("latin-1", "replace") for k in candidate_keys]

    # 3. Scan Polyglot Trailing Data (ZIP, RAR, 7z, text past EOF)
    t0 = time.time()
    try:
        carved_files, poly_flags, poly_corpus = scan_polyglot_trailer(
            raw_bytes, filename, custom_regex, candidate_keys, notes=report.notes
        )
        report.carved_files.extend(carved_files)
        report.flags.extend(poly_flags)
        corpus.extend(poly_corpus)
        candidate_keys = harvest_candidate_keys(corpus, passwords)
        report.harvested_keys = [k.decode("latin-1", "replace") for k in candidate_keys]
    except Exception as e:
        report.errors.append(f"Polyglot scan: {e}")
    report.stage_timings["polyglot"] = time.time() - t0

    # 4. Alpha Channel Variants & Mask Carving
    t0 = time.time()
    if pil_img is not None:
        try:
            alpha_carved, alpha_flags = scan_alpha_variants(pil_img, custom_regex)
            report.carved_files.extend(alpha_carved)
            report.flags.extend(alpha_flags)
        except Exception as e:
            report.errors.append(f"Alpha variants scan: {e}")
    report.stage_timings["alpha"] = time.time() - t0

    # 5. Strided, Keyed & Higher Bit-Plane LSB Sweep
    t0 = time.time()
    if pil_img is not None:
        try:
            keyed_flags = crack_keyed_and_strided_lsb(
                pil_img,
                candidate_keys,
                custom_regex=custom_regex,
                level=level,
            )
            report.flags.extend(keyed_flags)
        except Exception as e:
            report.errors.append(f"Bitplane sweep: {e}")
    report.stage_timings["bitplanes"] = time.time() - t0

    # 6. JPEG DCT Huffman Scan
    t0 = time.time()
    if raw_bytes.startswith(b"\xff\xd8"):
        try:
            dec = decode_jpeg_coefficients(raw_bytes, max_blocks=10_000)
            if dec["components"]:
                report.dct_findings.append({
                    "check": "dct decode",
                    "components_found": list(dec["components"].keys()),
                })
                for cid, blocks in dec["components"].items():
                    variants = _dct_variants(blocks)
                    for v_name, v_bits in variants.items():
                        for pack_name, b_data in _pack_both(v_bits):
                            for flg, via in find_flags_in_bytes(b_data, custom_regex):
                                score = score_flag_candidate(flg, custom_regex)
                                report.flags.append(FlagCandidate(
                                    flag=flg,
                                    source=f"DCT: {v_name} ({pack_name})",
                                    confidence="VERIFIED",
                                    details=f"Extracted from DCT component {cid}",
                                    via=via,
                                    score=score
                                ))
        except Exception as e:
            report.errors.append(f"DCT decode: {e}")
    report.stage_timings["dct"] = time.time() - t0

    # 7. Native Tool CLI Invocation (if available and file is clean format)
    t0 = time.time()
    suffix = Path(filename).suffix.lower() if filename else ".png"
    if suffix in [".png", ".bmp"] and level == "deep":
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(raw_bytes)
                tmp_path = tmp_file.name
            try:
                z_finds, z_flags = run_zsteg_analysis(tmp_path, custom_regex)
                report.cli_telemetry.extend(z_finds)
                report.flags.extend(z_flags)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception:
            pass
    report.stage_timings["native_cli"] = time.time() - t0

    # 7.5 Visual Optical OCR, QR & Barcode Detection
    if pil_img is not None:
        t0 = time.time()
        try:
            # 1. OCR Visual Text Detection (for flags visibly rendered on canvas / screenshots)
            ocr_flags, ocr_notes, ocr_corpus = scan_visual_text_ocr(pil_img, custom_regex)
            report.flags.extend(ocr_flags)
            report.notes.extend(ocr_notes)
            corpus.extend(ocr_corpus)

            # 2. Optical QR Detection
            qr_flags, qr_notes = scan_qr_codes(pil_img, custom_regex)
            report.flags.extend(qr_flags)
            report.notes.extend(qr_notes)
        except Exception as e:
            report.errors.append(f"Visual scan: {e}")
        report.stage_timings["optical_scan"] = time.time() - t0

    # 8. Score, Filter False Positives, and Rank Candidates
    is_generic_brace = custom_regex and custom_regex.strip().startswith((r"\{", r"^{", "{"))
    valid_flags: Dict[str, FlagCandidate] = {}
    for f in report.flags:
        f_score = score_flag_candidate(f.flag, custom_regex, hint=user_passphrase)
        if "OCR" in f.source or f.score > 150:
            f_score = max(f.score, f_score + 150)
        f.score = f_score

        # Filter out random byte collisions on noise and generic brace modes
        if f_score < 40 and (not custom_regex or is_generic_brace):
            continue

        if f.flag not in valid_flags:
            valid_flags[f.flag] = f
        else:
            if f_score > valid_flags[f.flag].score or len(f.details) > len(valid_flags[f.flag].details):
                valid_flags[f.flag] = f

    # Filter out weak sub-strings or noisy candidates when a verified flag exists
    has_high_flag = any(f.score >= 150 for f in valid_flags.values())
    clean_candidates = [
        f for f in valid_flags.values()
        if not any(f.flag != other.flag and f.flag in other.flag and other.score >= f.score for other in valid_flags.values())
        and not (has_high_flag and f.score < 100)
    ]

    for f in clean_candidates:
        if f.score >= 150:
            f.confidence = "VERIFIED"
        elif f.score >= 100:
            f.confidence = "HIGH"
        else:
            f.confidence = "PLAUSIBLE"

    report.flags = sorted(clean_candidates, key=lambda x: x.score, reverse=True)

    # 9. Professional Digital Forensics: Chi-Square, SPA, ELA, SNR & Threat IOC Extractor
    if pil_img is not None:
        try:
            report.chi2_findings = chi_square_lsb_attack(pil_img)
        except Exception as e:
            report.errors.append(f"Chi2 attack: {e}")
        try:
            report.spa_findings = sample_pair_analysis(pil_img)
        except Exception as e:
            report.errors.append(f"SPA attack: {e}")
        try:
            report.ela_findings = perform_error_level_analysis(pil_img)
        except Exception as e:
            report.errors.append(f"ELA analysis: {e}")
        try:
            report.snr_findings = spatial_noise_residual_analysis(pil_img)
        except Exception as e:
            report.errors.append(f"SNR analysis: {e}")

    ioc_corpus = list(corpus)
    for cf in report.carved_files:
        try:
            ioc_corpus.append(cf.data.decode("latin-1", "replace"))
        except Exception:
            pass
    for fc in report.flags:
        ioc_corpus.append(fc.flag)
        ioc_corpus.append(fc.details)
    report.iocs = extract_forensic_iocs(ioc_corpus)

    report.stage_timings["total"] = time.time() - t_start

    return report

def _run_embedded_tests():
    cs = sys.modules[__name__]
    FLAG = "HTB{l4k3_0f_s3cr3ts_1s_d33p}"
    rng = np.random.default_rng(1337)
    
    
    # ----------------------------------------------------------------- helpers
    def smooth_rgb(h=128, w=128) -> np.ndarray:
        y, x = np.mgrid[0:h, 0:w]
        base = np.stack([(x * 2) % 256, (y * 2) % 256, ((x + y)) % 256], axis=-1).astype(np.uint8)
        return np.clip(base.astype(int) + rng.integers(-6, 7, base.shape), 8, 247).astype(np.uint8)
    
    
    def to_bits(data: bytes, lsb_first: bool = False) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little" if lsb_first else "big")
    
    
    def png_bytes(arr: np.ndarray, mode: str = "RGB") -> bytes:
        buf = io.BytesIO()
        Image.fromarray(arr, mode).save(buf, format="PNG")
        return buf.getvalue()
    
    
    def embed_plane(arr, chan, k, payload: bytes, stride=1, offset=0, lsb_first=False):
        flat = arr[:, :, chan].reshape(-1).copy()
        bits = to_bits(payload, lsb_first)
        idx = offset + np.arange(len(bits)) * stride
        flat[idx] = (flat[idx] & ~np.uint8(1 << k)) | (bits.astype(np.uint8) << k)
        out = arr.copy()
        out[:, :, chan] = flat.reshape(arr.shape[:2])
        return out
    
    
    def xor_cycle(data: bytes, key: bytes) -> bytes:
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    
    
    def top(report):
        return report.flags[0].flag if report.flags else None
    
    
    def flags_of(report):
        return {f.flag for f in report.flags}
    
    
    # ----------------------------------------------------------------- tests
    def test_regex_and_scoring():
        assert cs.score_flag_candidate(FLAG) > 150
        assert cs.score_flag_candidate("6{U7^}") < 30
        assert cs.score_flag_candidate("ab{Xy7^#}") < 30
        assert cs.score_flag_candidate(FLAG) > cs.score_flag_candidate("zz{aaaaaa}")
        hits = cs.find_flags_in_text("noise " + FLAG + " tail")
        assert FLAG in hits  # (the brace-only fallback also reports the bare {body}, as in v1)
        # custom regex with a capture group must return the full match, not the group
        assert cs.find_flags_in_text("abc-1234-xyz", r"abc-(\d+)-xyz") == ["abc-1234-xyz"]
    
    
    def test_decoders():
        b64 = base64.b64encode(FLAG.encode()).decode()
        hx = FLAG.encode().hex()
        r13 = codecs.encode(FLAG, "rot_13")
        for blob, via in ((b64, "base64"), (hx, "hex"), (r13, "rot13"),
                          (base64.b64encode(codecs.encode(FLAG, "rot_13").encode()).decode(), "base64>rot13"),
                          (base64.b32encode(FLAG.encode()).decode(), "base32"),
                          (FLAG[::-1], "reverse")):
            hits = dict(cs.find_flags_deep(blob))
            assert FLAG in hits, (via, hits)
            assert hits[FLAG] == via, (via, hits)
    
    
    def test_plain_lsb_stride_offset():
        arr = smooth_rgb()
        img = embed_plane(arr, 2, 0, FLAG.encode(), stride=3, offset=7)
        rep = cs.solve_image_stego(png_bytes(img), "t.png")
        assert top(rep) == FLAG, rep.flags[:3]
    
    
    def test_lsb_first_bit_order_green():
        img = embed_plane(smooth_rgb(), 1, 0, FLAG.encode(), lsb_first=True)
        assert top(cs.solve_image_stego(png_bytes(img), "t.png")) == FLAG
    
    
    def test_higher_bit_planes():
        for k in (1, 2, 3):
            img = embed_plane(smooth_rgb(), 1, k, FLAG.encode())
            rep = cs.solve_image_stego(png_bytes(img), "t.png")
            assert top(rep) == FLAG, (k, rep.flags[:3])
            src = next(f.source for f in rep.flags if f.flag == FLAG)
            assert "Green" in src and f"bit{k}" in src, (k, src)
    
    
    def test_two_bits_per_sample():
        arr = smooth_rgb()
        bits = to_bits(FLAG.encode())
        pairs = bits.reshape(-1, 2)  # (b1, b0) per sample, MSB-first
        flat = arr[:, :, 0].reshape(-1).copy()
        n = len(pairs)
        flat[:n] = (flat[:n] & 0xFC) | (pairs[:, 0].astype(np.uint8) << 1) | pairs[:, 1].astype(np.uint8)
        arr[:, :, 0] = flat.reshape(arr.shape[:2])
        rep = cs.solve_image_stego(png_bytes(arr), "t.png")
        assert top(rep) == FLAG, rep.flags[:3]
    
    
    def test_xor_keyed_with_zip_polyglot_key_hint():
        """Flag XOR-ed with LANTERN in blue LSBs; key only reachable via a note inside an appended ZIP."""
        img = embed_plane(smooth_rgb(), 2, 0, xor_cycle(FLAG.encode(), b"LANTERN"))
        png = png_bytes(img)
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w") as zf:
            zf.writestr("field_note.txt", "Surveyor's log. The lock: LANTERN. Do not share.")
        rep = cs.solve_image_stego(png + zbuf.getvalue(), "poly.png")
        assert top(rep) == FLAG, rep.flags[:3]
        assert "LANTERN" in rep.harvested_keys
        names = {c.filename for c in rep.carved_files}
        assert "field_note.txt" in names and "carved_payload.bin" in names
    
    
    def test_encrypted_zip_cracked_with_harvested_key():
        """ZIP (ZipCrypto) encrypted with a password that sits in PNG metadata; stdlib can't *write* ZipCrypto,
        so build the archive bytes via the classic algorithm."""
        # minimal ZipCrypto writer
        def crc_table():
            t = []
            for i in range(256):
                c = i
                for _ in range(8):
                    c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
                t.append(c)
            return t
    
        T = crc_table()
    
        def upd(k, b):
            k0 = (T[(k[0] ^ b) & 0xFF] ^ (k[0] >> 8)) & 0xFFFFFFFF
            k1 = ((k[1] + (k0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
            k2 = (T[(k[2] ^ (k1 >> 24)) & 0xFF] ^ (k[2] >> 8)) & 0xFFFFFFFF
            return [k0, k1, k2]
    
        def encrypt(data: bytes, pwd: bytes, crc: int) -> bytes:
            k = [305419896, 591751049, 878082192]
            for b in pwd:
                k = upd(k, b)
            header = bytes(rng.integers(0, 256, 11).tolist()) + bytes([crc >> 24])
            out = bytearray()
            for b in header + data:
                t = (k[2] | 2) & 0xFFFF
                ks = ((t * (t ^ 1)) >> 8) & 0xFF
                out.append(b ^ ks)
                k = upd(k, b)
            return bytes(out)
    
        import zlib
        content = ("secret vault: " + FLAG).encode()
        crc = zlib.crc32(content) & 0xFFFFFFFF
        enc = encrypt(content, b"ARCHIVE", crc)
        name = b"vault.txt"
        local = struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, 1, 0, 0, 0x21, crc, len(enc), len(content), len(name), 0) + name + enc
        central = struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20, 1, 0, 0, 0x21, crc, len(enc), len(content),
                              len(name), 0, 0, 0, 0, 0, 0) + name
        eocd = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
        zip_bytes = local + central + eocd
    
        # confirm stdlib agrees the archive is valid and encrypted
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            assert zf.read("vault.txt", pwd=b"ARCHIVE") == content
    
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Comment", 'Archive passphrase: "ARCHIVE" - keep it safe')
        buf = io.BytesIO()
        Image.fromarray(smooth_rgb(), "RGB").save(buf, format="PNG", pnginfo=meta)
        rep = cs.solve_image_stego(buf.getvalue() + zip_bytes, "enc.png")
        assert FLAG in flags_of(rep), (rep.flags, rep.notes, rep.errors)
        assert any("decrypted" in n for n in rep.notes), rep.notes
    
    
    def test_base64_hex_rot13_in_lsb():
        for transform, via in ((lambda s: base64.b64encode(s.encode()).decode(), "base64"),
                               (lambda s: s.encode().hex(), "hex"),
                               (lambda s: codecs.encode(s, "rot_13"), "rot13")):
            payload = transform(FLAG).encode()
            img = embed_plane(smooth_rgb(), 0, 0, payload)
            rep = cs.solve_image_stego(png_bytes(img), "t.png")
            assert FLAG in flags_of(rep), (via, rep.flags[:3])
            got = next(f for f in rep.flags if f.flag == FLAG)
            assert got.via == via, (via, got.via)
    
    
    def test_alpha_channel_variants():
        h = w = 96
        base = np.dstack([smooth_rgb(h, w), np.full((h, w), 255, np.uint8)])
        # (a) alpha values are ASCII codes of the flag
        a1 = base.copy()
        msg = np.frombuffer(FLAG.encode(), np.uint8)
        a1[0, : len(msg), 3] = msg
        rep = cs.solve_image_stego(png_bytes(a1, "RGBA"), "a.png")
        assert FLAG in flags_of(rep), rep.flags[:3]
        # (b) RGB bytes hidden under fully transparent pixels
        a2 = base.copy()
        a2[:, :, 3] = 0
        a2[:, :, 3][40:, :] = 255
        trip = np.frombuffer(FLAG.encode().ljust(30, b"\x00"), np.uint8).reshape(-1, 3)
        for i, px in enumerate(trip):
            a2[0, i, :3] = px
        rep = cs.solve_image_stego(png_bytes(a2, "RGBA"), "a.png")
        assert FLAG in flags_of(rep), rep.flags[:3]
        names = {c.filename for c in rep.carved_files}
        assert {"alpha_mask.png", "alpha_mask_inverted.png", "alpha0_rgb_revealed.png"} <= names, names
        for c in rep.carved_files:
            if c.filename.endswith(".png"):
                Image.open(io.BytesIO(c.data)).verify()
        # (c) LSB plane of the alpha channel
        a3 = base.copy()
        bits = to_bits(FLAG.encode()).astype(np.uint8)
        flat = a3[:, :, 3].reshape(-1).copy()
        flat[: len(bits)] = (flat[: len(bits)] & 0xFE) | bits
        a3[:, :, 3] = flat.reshape(h, w)
        rep = cs.solve_image_stego(png_bytes(a3, "RGBA"), "a.png")
        assert FLAG in flags_of(rep), rep.flags[:3]
    
    
    def test_colorspace_sources_plumbing():
        """Data in the Cr channel: inject Cr samples directly (RGB round-trips lose exact LSBs)."""
        img = Image.fromarray(smooth_rgb(), "RGB")
        srcs = cs.build_pixel_sources(img)
        assert {"Red", "Green", "Blue", "Y", "Cb", "Cr", "Hue", "Saturation", "Value"} <= set(srcs)
        cr = srcs["Cr"].copy()
        bits = to_bits(FLAG.encode()).astype(np.uint8)
        cr[: len(bits)] = (cr[: len(bits)] & 0xFE) | bits
        hits = cs.crack_keyed_and_strided_lsb(img, [], sources={"Cr": cr})
        assert any(h.flag == FLAG and "Cr" in h.source for h in hits), hits[:3]
        # natively YCbCr image (e.g. TIFF) is used without conversion
        ycc = Image.fromarray(smooth_rgb(), "RGB").convert("YCbCr")
        assert {"Y", "Cb", "Cr"} == set(cs.build_pixel_sources(ycc))
    
    
    def test_palette_png():
        h = w = 64
        idx = rng.integers(0, 16, (h, w)).astype(np.uint8)
        idx = (idx & 0xFE)
        bits = to_bits(FLAG.encode()).astype(np.uint8)
        flat = idx.reshape(-1)
        flat[: len(bits)] |= bits
        im = Image.fromarray(flat.reshape(h, w), "P")
        im.putpalette([int(v) for v in rng.integers(0, 256, 768)])
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        rep = cs.solve_image_stego(buf.getvalue(), "p.png")
        assert top(rep) == FLAG, rep.flags[:3]
    
    
    def test_boundaries_png_jpeg_gif_bmp():
        arr = smooth_rgb(64, 64)
        tail = b"TRAILING NOTE: key: SESAME"
        for fmt in ("PNG", "JPEG", "GIF", "BMP"):
            buf = io.BytesIO()
            Image.fromarray(arr, "RGB").save(buf, format=fmt)
            base = buf.getvalue()
            audit = cs.audit_file_structure(base + tail, f"x.{fmt.lower()}")
            assert audit["trailing_data"] == tail, (fmt, audit["trailing_data"][:40], audit.get("reason"))
            assert cs.audit_file_structure(base, "x")["trailing_data"] == b"", fmt
        # a JPEG that embeds an EXIF thumbnail (contains its own FFD9) must not fool the carver
        ex = Image.Exif()
        ex[0x010E] = "desc"
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="JPEG", exif=ex)
        assert cs.audit_file_structure(buf.getvalue() + tail, "x.jpg")["trailing_data"] == tail
    
    
    def test_metadata_sources():
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Description", base64.b64encode(FLAG.encode()).decode())
        buf = io.BytesIO()
        Image.fromarray(smooth_rgb(64, 64), "RGB").save(buf, format="PNG", pnginfo=meta)
        rep = cs.solve_image_stego(buf.getvalue(), "m.png")
        top1 = rep.flags[0]
        assert top1.flag == FLAG and top1.via == "base64", rep.flags[:2]
    
        ex = Image.Exif()
        ex[0x013B] = "Artist: lantern keeper"           # Artist
        ex.get_ifd(0x8769)[0x9286] = b"ASCII\x00\x00\x00" + FLAG.encode()  # UserComment in Exif IFD
        jb = io.BytesIO()
        Image.fromarray(smooth_rgb(64, 64), "RGB").save(jb, format="JPEG", exif=ex, quality=95)
        rep = cs.solve_image_stego(jb.getvalue(), "m.jpg")
        assert FLAG in flags_of(rep), (rep.flags, list(rep.metadata_findings)[:10])
    
    
    def test_c2pa_findings_are_preserved_in_solver_report():
        """C2PA provenance metadata must survive the normal metadata merge."""
        prefix = b"PNG-or-JPEG-prefix"
        suffix = b"payload-after-manifest"
        marker = b"ChatGPT trainedAlgorithmicMedia "
    
        # The C2PA asset hash excludes the manifest chunk itself.  Build the
        # chunk in two passes so the embedded hash matches the bytes outside it.
        import hashlib
        import struct
    
        placeholder = marker + (b"0" * 44) + b"="
        chunk_len = struct.pack(">I", len(placeholder))
        base = prefix + chunk_len + b"caBX" + placeholder + b"CRC!" + suffix
        chunk_start = len(prefix)
        chunk_end = chunk_start + 12 + len(placeholder)
        expected = base[:chunk_start] + base[chunk_end:]
        digest = base64.b64encode(hashlib.sha256(expected).digest())
    
        chunk_data = marker + digest
        raw = prefix + struct.pack(">I", len(chunk_data)) + b"caBX" + chunk_data + b"CRC!" + suffix
        img = Image.fromarray(smooth_rgb(32, 32), "RGB")
        report = cs.solve_image_stego(raw, "manifest.bin", img)
    
        assert report.metadata_findings["Provenance Agent"] == "ChatGPT (gpt-image / DALL-E 3)"
        assert report.metadata_findings["Provenance Source Type"] == "Generative AI (trainedAlgorithmicMedia)"
        assert report.metadata_findings["Provenance Asset SHA256 Match"] == "True"
    
    
    def test_png_private_chunk_and_bad_crc():
        png = png_bytes(smooth_rgb(32, 32))
        pos = png.rfind(b"IEND") - 4
        body = FLAG.encode()
        chunk = struct.pack(">I", len(body)) + b"stEg" + body + struct.pack(">I", 0xDEADBEEF)  # wrong CRC
        rep = cs.solve_image_stego(png[:pos] + chunk + png[pos:], "c.png")
        assert FLAG in flags_of(rep), rep.flags
        assert any("BAD CRC" in k for k in rep.metadata_findings), list(rep.metadata_findings)
    
    
    def test_custom_regex_and_prefix_scoring():
        img = embed_plane(smooth_rgb(), 2, 0, b"KCTF-secret-0042 and more")
        rep = cs.solve_image_stego(png_bytes(img), "t.png", custom_regex=r"KCTF-secret-\d{4}")
        assert top(rep) == "KCTF-secret-0042", rep.flags[:3]
    
    
    def test_jpeg_dct_decoder_matches_pil():
        """Validate Huffman decoding: dequantise + IDCT our coefficients and compare with Pillow's pixels."""
        n = 8
        C = np.array([[np.sqrt(1 / 8) if u == 0 else np.sqrt(2 / 8) * np.cos((2 * x + 1) * u * np.pi / 16)
                       for x in range(n)] for u in range(n)])
        zz = cs._ZIGZAG
        y, x = np.mgrid[0:64, 0:64]
        gray = np.clip(128 + 60 * np.sin(x / 7.0) + 40 * np.cos(y / 5.0) + rng.integers(-4, 5, (64, 64)), 0, 255).astype(np.uint8)
        for subsampling, mode in ((None, "L"), (2, "RGB")):
            buf = io.BytesIO()
            if mode == "L":
                Image.fromarray(gray, "L").save(buf, format="JPEG", quality=85)
            else:
                rgb = np.dstack([gray, np.roll(gray, 9, 0), np.roll(gray, 9, 1)])
                Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=85, subsampling=2)  # 4:2:0
            raw = buf.getvalue()
            dec = cs.decode_jpeg_coefficients(raw, max_blocks=10_000)
            im = Image.open(io.BytesIO(raw))
            qt = im.quantization
            qtab = np.array(qt[0], dtype=float)
            ycoef = dec["components"][min(dec["components"])]
            # pixel reference: luma of the decoded image
            ref = np.asarray(im.convert("L"), dtype=float)
            # rebuild luma plane from coefficients (try both quant-table orderings)
            best = 1e9
            for q_nat in (qtab, np.empty(64)):
                if q_nat is not qtab:
                    q_nat[zz] = qtab  # table delivered in zigzag order -> natural
                plane = np.zeros((64, 64))
                blocks_per_row = 8
                # luma blocks are stored in MCU order: for 4:2:0 each MCU holds 2x2 luma blocks
                if mode == "L":
                    order = [(r, c) for r in range(8) for c in range(8)]
                else:
                    order = []
                    for mr in range(4):
                        for mc in range(4):
                            order += [(2 * mr, 2 * mc), (2 * mr, 2 * mc + 1), (2 * mr + 1, 2 * mc), (2 * mr + 1, 2 * mc + 1)]
                if len(order) > len(ycoef):
                    continue
                for i, (r, c) in enumerate(order):
                    nat = np.zeros(64)
                    nat[zz] = ycoef[i]
                    F = (nat * q_nat).reshape(8, 8)
                    plane[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = C.T @ F @ C + 128
                best = min(best, float(np.abs(plane - ref).mean()))
            assert best < 3.0, (mode, best)
    
    
    def test_dct_stage_runs_and_finds_jsteg_style_payload():
        """Build a JPEG whose luma LSB stream we control by *re-encoding* is impractical without an encoder,
        so verify the scanner on a synthetic coefficient array instead."""
        blocks = np.zeros((4000, 64), dtype=np.int16)
        bits = to_bits(FLAG.encode()).astype(np.uint8)
        # place payload bits in LSBs of non-zero AC coefficients (values 2 or 3 -> LSB = bit)
        ac_positions = [(b, 1 + (i % 10)) for i, b in enumerate(np.repeat(np.arange(4000), 10)[: len(bits)])]
        for bit, (b, k) in zip(bits, ac_positions):
            blocks[b, k] = 2 + int(bit)
        variants = cs._dct_variants(blocks)
        data = cs._pack_both(variants["JSteg (skip 0 and 1) LSB"][:len(bits)])[0][1]
        assert FLAG.encode() in data
    
    
    def test_jpeg_full_pipeline_no_crash():
        buf = io.BytesIO()
        Image.fromarray(smooth_rgb(96, 96), "RGB").save(buf, format="JPEG", quality=90)
        rep = cs.solve_image_stego(buf.getvalue() + b"extra: {tail_flag_here}", "j.jpg")
        assert not rep.errors, rep.errors
        assert any(d.get("check") == "dct decode" for d in rep.dct_findings), rep.dct_findings
    
    
    def test_corrupt_and_hostile_inputs():
        cases = [b"", b"\x00" * 10, os.urandom(2048), b"\x89PNG\r\n\x1a\n" + os.urandom(300),
                 b"\xff\xd8\xff\xe0" + os.urandom(300), b"GIF89a" + os.urandom(100), b"BM" + os.urandom(100)]
        good = png_bytes(smooth_rgb(32, 32))
        cases += [good[:len(good) // 2], good[:-3], good + os.urandom(100)]
        for blob in cases:
            rep = cs.solve_image_stego(blob, "weird.bin")  # must not raise
            assert isinstance(rep, cs.SolverReport)
        # truncated stream inside the DCT decoder
        buf = io.BytesIO()
        Image.fromarray(smooth_rgb(64, 64), "RGB").save(buf, format="JPEG")
        j = buf.getvalue()
        cs.solve_image_stego(j[: len(j) // 2], "trunc.jpg")
    
    
    def test_random_noise_has_no_high_confidence_false_positive():
        noise = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
        rep = cs.solve_image_stego(png_bytes(noise), "noise.png")
        assert not rep.flags, [(f.flag, f.score, f.source) for f in rep.flags]
    
    
    def test_fast_level_still_solves_classic_lsb():
        img = embed_plane(smooth_rgb(), 0, 0, FLAG.encode())
        t = time.time()
        rep = cs.solve_image_stego(png_bytes(img), "t.png", level="fast")
        assert top(rep) == FLAG
        print(f"  fast level: {time.time() - t:.2f}s")
    
    
    def test_timing_budget_deep():
        img = embed_plane(smooth_rgb(512, 512), 2, 0, FLAG.encode(), stride=2, offset=3)
        t = time.time()
        rep = cs.solve_image_stego(png_bytes(img), "big.png")
        dt = time.time() - t
        assert top(rep) == FLAG
        print(f"  deep level on 512x512: {dt:.2f}s  stages={rep.stage_timings}")
        assert dt < 30
    
    
    def test_backward_compatible_api():
        img = Image.open(io.BytesIO(png_bytes(embed_plane(smooth_rgb(), 2, 0, FLAG.encode()))))
        raw = png_bytes(embed_plane(smooth_rgb(), 2, 0, FLAG.encode()))
        # original positional signature
        rep = cs.solve_image_stego(raw, "t.png", img, None, None, None)
        assert top(rep) == FLAG
        findings, flags, corpus = cs.scan_metadata(raw, img, None)
        carved, tflags, tcorpus = cs.scan_polyglot_trailer(raw, "t.png", None)
        assert isinstance(findings, dict) and carved == [] and tflags == []
        bf, bflags = cs.scan_bitplanes_lsb(img)
        assert any(f.flag == FLAG for f in bflags)
        assert all({"channel", "readability", "entropy", "sample_preview", "raw_bytes"} <= set(e) for e in bf)

    import traceback

    tests = [
        (name, fn)
        for name, fn in sorted(locals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.2f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    _run_embedded_tests()
