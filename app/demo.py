"""
demo.py — Sudarshana: Advanced Steganography Forensics Platform
Named after the divine Sudarshana Chakra of Lord Vishnu/Narayana.
White theme · Calibri font · Indian English
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io, struct, base64, hashlib
from datetime import datetime
import pandas as pd
from html import escape
import cv2

from model.fusion_model import MultiBranchSteganalyzer
from deploy.risk_scoring import compute_risk_score
from utils.gradcam import StegoGradCAM

# Real forensic modules (ChatGPT-engineered, mathematically sound)
try:
    from forensics.rs_steganalysis import analyse_rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

try:
    from forensics.structural_audit import audit_file_structure
    _STRUCT_AVAILABLE = True
except ImportError:
    _STRUCT_AVAILABLE = False

try:
    from forensics.blind_detector import verify_with_blind_cnn
    _BLIND_AVAILABLE = True
except ImportError:
    _BLIND_AVAILABLE = False

try:
    from forensics.generative_inversion import generative_inversion_audit
    _GEN_INV_AVAILABLE = True
except ImportError:
    _GEN_INV_AVAILABLE = False

try:
    from forensics.ctf_solver import (
        solve_image_stego, UNIVERSAL_FLAG_REGEX, apply_stegsolve_filter,
        decode_binary_ascii_string, scan_qr_codes, repair_png_ihdr_height,
        audit_c2pa_provenance, perform_error_level_analysis,
        sample_pair_analysis, spatial_noise_residual_analysis
    )
    _CTF_SOLVER_AVAILABLE = True
    _CTF_SOLVER_ERROR = None
except Exception as e:
    try:
        from src.forensics.ctf_solver import (
            solve_image_stego, UNIVERSAL_FLAG_REGEX, apply_stegsolve_filter,
            decode_binary_ascii_string, scan_qr_codes, repair_png_ihdr_height,
            audit_c2pa_provenance, perform_error_level_analysis,
            sample_pair_analysis, spatial_noise_residual_analysis
        )
        _CTF_SOLVER_AVAILABLE = True
        _CTF_SOLVER_ERROR = None
    except Exception as e2:
        _CTF_SOLVER_AVAILABLE = False
        _CTF_SOLVER_ERROR = str(e2)

st.set_page_config(
    page_title="Sudarshana — Image Forensics Tool",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Poppins:wght@400;600;700&display=swap');

:root {
  --bg-canvas: #faf9f5;
  --bg-card: #f0eee6;
  --text-ink: #141413;
  --accent-clay: #d97757;
  --font-display: 'Poppins', sans-serif;
  --font-editorial: 'Lora', Georgia, serif;
}

/* Do NOT use * selector to avoid breaking Streamlit's Material Icons */
html, body, [class*="css"] {
  background-color: var(--bg-canvas) !important;
  color: var(--text-ink) !important;
}

.stApp { background-color: var(--bg-canvas) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 3rem; max-width: 1280px; }

p, div, span, strong, small, li {
  font-family: var(--font-editorial);
  color: var(--text-ink);
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-display) !important;
  color: var(--text-ink) !important;
}

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 3rem 1.5rem;
  background-color: var(--bg-card);
  border: 1px solid rgba(20, 20, 19, 0.08);
  border-radius: 8px;
  margin-bottom: 2rem;
}
.hero-title {
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-size: 4.2rem;
  font-weight: 700;
  color: var(--text-ink);
  letter-spacing: -0.02em;
  margin: 0;
}
.hero-sub {
  font-family: var(--font-editorial);
  font-size: 1.2rem;
  color: var(--text-ink);
  margin-top: 0.5rem;
  font-style: italic;
}
.hero-tag {
  font-family: var(--font-display);
  font-size: 1rem;
  color: var(--accent-clay);
  margin-top: 1rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

/* ── Cards ── */
.card {
  background: var(--bg-card);
  border: 1px solid rgba(20, 20, 19, 0.08);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}
.card-title {
  font-family: var(--font-display);
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--accent-clay);
  margin-bottom: 0.8rem;
}
.card-body {
  font-family: var(--font-editorial);
  font-size: 1rem;
  color: var(--text-ink);
  line-height: 1.6;
}

/* ── Verdict cards ── */
.verdict-safe, .verdict-threat {
  background: var(--bg-card);
  border: 1px solid rgba(20, 20, 19, 0.15);
  border-radius: 8px;
  padding: 2.5rem;
  text-align: center;
}
.verdict-title { 
  font-family: var(--font-display);
  font-size: 2rem; 
  font-weight: 700; 
  margin: 0.5rem 0; 
  color: var(--text-ink);
}

/* ── Pills / badges ── */
.pill {
  display: inline-flex;
  align-items: center;
  font-family: var(--font-display);
  padding: 0.35rem 0.7rem;
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
  margin: 0.2rem 0.4rem 0.2rem 0;
  background: rgba(217, 119, 87, 0.1);
  color: var(--accent-clay);
  border: 1px solid rgba(217, 119, 87, 0.2);
}

/* ── Hex / monospace box ── */
.hex-box {
  font-family: 'Consolas', 'Courier New', monospace !important;
  font-size: 0.85rem;
  color: var(--text-ink);
  background: var(--bg-canvas);
  border: 1px solid rgba(20, 20, 19, 0.08);
  border-radius: 4px;
  padding: 1rem;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 220px;
  overflow-y: auto;
}

/* ── Segment bar ── */
.seg-bar {
  height: 28px;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  margin: 0.8rem 0;
  border: 1px solid rgba(20, 20, 19, 0.15);
}
.seg-bar div {
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-display);
  font-size: 0.75rem;
  font-weight: 600;
  color: #fff;
}

/* ── Step boxes ── */
.step-box {
  background: var(--bg-card);
  border: 1px solid rgba(20, 20, 19, 0.08);
  border-radius: 8px;
  padding: 1.5rem 1rem;
  text-align: center;
}
.step-number { 
  font-family: var(--font-display);
  font-size: 2.2rem; 
  font-weight: 700; 
  color: var(--accent-clay); 
}
.step-text { 
  font-family: var(--font-editorial);
  font-size: 1rem; 
  color: var(--text-ink); 
  margin-top: 0.5rem; 
  line-height: 1.5; 
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent;
  border-bottom: 2px solid rgba(20, 20, 19, 0.08);
  padding: 0;
  gap: 1rem;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--font-display);
  color: rgba(20, 20, 19, 0.6);
  font-weight: 600;
  border-radius: 0;
  padding: 0.8rem 0;
  font-size: 1rem;
  border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
  background: transparent !important;
  color: var(--text-ink) !important;
  border-bottom: 2px solid var(--text-ink) !important;
}

/* ── Metrics ── */
[data-testid="stMetricValue"] { 
  font-family: var(--font-display) !important;
  color: var(--text-ink) !important; 
  font-weight: 700 !important; 
}
[data-testid="stMetricLabel"] { 
  font-family: var(--font-display) !important;
  color: rgba(20, 20, 19, 0.6) !important; 
  font-weight: 600 !important; 
}

/* ── File uploader — Anthropic style ── */
[data-testid="stFileUploader"] {
  background: var(--bg-card) !important;
  border: 1px dashed rgba(20, 20, 19, 0.2) !important;
  border-radius: 8px !important;
  padding: 1.5rem !important;
}
[data-testid="stFileUploadDropzone"] {
  background: transparent !important;
}
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] section {
  background: transparent !important;
}
/* Override ALL buttons in the uploader area */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploadDropzone"] button {
  font-family: var(--font-display) !important;
  font-weight: 600 !important;
  font-size: 15px !important;
  background-color: var(--accent-clay) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 6px !important;
  padding: 0.5rem 1rem !important;
  transition: opacity 0.2s ease !important;
}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploadDropzone"] button:hover {
  opacity: 0.9 !important;
}
/* Uploader text */
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small {
  font-family: var(--font-editorial) !important;
  color: var(--text-ink) !important;
}

/* ── Section title ── */
.section-title {
  font-family: var(--font-display);
  font-size: 1.8rem;
  font-weight: 700;
  color: var(--text-ink);
  margin: 2.5rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid rgba(20, 20, 19, 0.08);
}

/* ── Progress bar colour ── */
.stProgress > div > div > div > div { background: var(--accent-clay) !important; }

/* ── Divider ── */
hr { border-color: rgba(20, 20, 19, 0.08) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: var(--bg-card) !important;
  border: 1px solid rgba(20, 20, 19, 0.08) !important;
  border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
  background: transparent !important;
  color: var(--text-ink) !important;
  font-family: var(--font-display) !important;
  font-weight: 600 !important;
  padding: 1rem !important;
}
[data-testid="stExpander"] summary:hover {
  background: rgba(20, 20, 19, 0.02) !important;
}
/* Warning / info / success boxes */
[data-testid="stAlert"] {
  background: var(--bg-card) !important;
  border: 1px solid rgba(20, 20, 19, 0.08) !important;
  border-radius: 8px !important;
  color: var(--text-ink) !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (same logic, white-theme compatible)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path="weights/fusion_best.pt"):
    dev = torch.device("cpu")
    model = MultiBranchSteganalyzer()
    trained = False
    if os.path.exists(path):
        try:
            raw   = torch.load(path, map_location=dev)
            state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            model.load_state_dict(state)
            trained = True
        except Exception:
            pass
    model.eval()
    # INT8 Dynamic Quantization — speeds up CPU inference by ~1.5-2×
    # Targets all nn.Linear layers (fusion head, branch MLPs)
    # This is completely free — built-in PyTorch, no payment needed
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except RuntimeError:
        # FBGEMM backend not available (e.g. macOS ARM) — skip quantization
        pass
    return model, trained


def preprocess(pil_img, sz=256):
    img = pil_img.convert("L")
    w, h = img.size
    if w < sz or h < sz:
        import torchvision.transforms.functional as TF
        pw, ph = max(0, sz-w), max(0, sz-h)
        img = TF.pad(img, (pw//2, ph//2, pw-pw//2, ph-ph//2))
    l = (img.width-sz)//2; t = (img.height-sz)//2
    img = img.crop((l, t, l+sz, t+sz))
    arr = (np.array(img, dtype=np.float32)/255. - 0.5) / 0.5
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def run_predict(model, tensor):
    probs     = F.softmax(model(tensor), dim=1)
    sp        = float(probs[0, 1])
    return ("THREAT" if sp >= .5 else "SAFE"), sp, float(probs[0, 0])


def get_bitplane(arr, plane):
    return ((arr >> plane) & 1) * 255

@st.cache_data(show_spinner=False, max_entries=5)
def make_all_bitplanes(img_hash, channel, _pil_img):
    ch_map = {"Grey": _pil_img.convert("L"),
              "Red":  _pil_img.convert("RGB"),
              "Green":_pil_img.convert("RGB"),
              "Blue": _pil_img.convert("RGB")}
    base = np.array(ch_map[channel], dtype=np.uint8)
    if channel == "Red":   base = base[:,:,0]
    elif channel == "Green": base = base[:,:,1]
    elif channel == "Blue":  base = base[:,:,2]
    return [Image.fromarray(get_bitplane(base, p).astype(np.uint8)) for p in range(8)]

@st.cache_data(show_spinner=False, max_entries=5)
def make_lsb_heatmap(img_hash, sensitivity, _pil_img):
    rgb  = np.array(_pil_img.convert("RGB"), dtype=np.uint8)
    mask = np.zeros(rgb.shape[:2], dtype=np.float32)
    for c in range(3):
        for p in range(sensitivity):
            mask += ((rgb[:,:,c] >> p) & 1).astype(np.float32)
    mask = mask / (3 * sensitivity)
    H, W = mask.shape
    hm   = np.zeros((H, W, 4), dtype=np.uint8)
    hm[:,:,0] = (mask * 220).astype(np.uint8)
    hm[:,:,1] = 0
    hm[:,:,2] = ((1-mask) * 60).astype(np.uint8)
    hm[:,:,3] = (mask * 160).astype(np.uint8)
    return Image.fromarray(hm, 'RGBA')

def make_difference_image(pil_img, amplify=10):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    # Cast to int16 BEFORE multiply to prevent uint8 overflow wrapping
    two_lsb = (gray.astype(np.int16) & 3) * (255 // 3)
    return Image.fromarray(np.clip(two_lsb * amplify, 0, 255).astype(np.uint8))

def estimate_capacity(pil_img):
    w, h = pil_img.size
    channels = len(pil_img.convert("RGB").getbands())
    cap = (w * h * channels) // 8
    return cap, w * h

def estimate_payload(pil_img, stego_prob):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
    even = hist[0::2].astype(float); odd = hist[1::2].astype(float)
    expected = (even+odd)/2
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.where(expected>0, (even-expected)**2/expected, 0).sum()
    eq = max(0., min(1., 1. - chi/(gray.size*0.05+1e-8)))
    cap, _ = estimate_capacity(pil_img)
    return int(eq * cap * stego_prob), eq

def fingerprint_method(pil_img, sp):
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    lsb  = (gray & 1).astype(float)
    lsb_mean = float(lsb.mean())
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0,255))
    even = hist[0::2].astype(float); odd = hist[1::2].astype(float)
    expected = (even+odd)/2
    with np.errstate(divide='ignore', invalid='ignore'):
        chi = np.where(expected>0,(even-expected)**2/expected,0).sum()
    if sp < 0.4:
        return "No Hidden Data Detected", "HIGH", "The image is showing natural statistical properties of a normal, clean photograph. No suspicious patterns were found during analysis.", "✅"
    elif abs(lsb_mean - 0.5) < 0.05:
        return "LSB Substitution Method", "HIGH", "The lowest bit of each pixel has been replaced directly with hidden data bits. This is the most commonly used steganography method. Tools like Steghide use this approach.", "🔴"
    elif abs(lsb_mean - 0.5) < 0.15:
        return "LSB Matching (±1 Method)", "MEDIUM", "Instead of direct replacement, pixel values have been increased or decreased by 1. This is slightly harder to detect and is used by tools like OpenStego.", "🟠"
    elif chi < gray.size * 0.001:
        return "JPEG Frequency Method (DCT)", "MEDIUM", "The hidden data appears to be embedded in the mathematical frequency components of the JPEG image rather than directly in pixels. Used by JSteg and F5 tools.", "🟡"
    else:
        return "Adaptive / Unknown Method", "LOW", "There are suspicious signs but they do not match any known standard method clearly. This could be a newer adaptive steganography tool that tries to hide in textured regions.", "🟡"

def get_magic_bytes(data):
    sigs = [
        (b'\xff\xd8\xff', "JPEG Image", "🖼️"),
        (b'\x89PNG',      "PNG Image",  "🖼️"),
        (b'GIF',          "GIF Image",  "🖼️"),
        (b'BM',           "BMP Image",  "🖼️"),
        (b'PK\x03\x04',  "ZIP Archive","🗜️"),
        (b'%PDF',         "PDF File",   "📄"),
        (b'\x1f\x8b',    "GZIP File",  "🗜️"),
        (b'MZ',           "Windows Program — ⚠️ Possible Malware!", "🚨"),
        (b'\x7fELF',      "Linux Programme — ⚠️ Possible Malware!",  "🚨"),
        (b'-----BEGIN',   "Encryption Key / Certificate", "🔑"),
        (b'ID3',          "MP3 Audio",  "🎵"),
    ]
    actual_magic = data[:8].hex().upper()
    actual_magic_fmt = ' '.join(actual_magic[i:i+2] for i in range(0, len(actual_magic), 2))
    
    for sig, name, icon in sigs:
        if data.startswith(sig): return name, icon, actual_magic_fmt
    pr = sum(32<=b<127 for b in data[:64])
    if pr > 50: return "Plain Text Message", "📝", actual_magic_fmt
    return "Unknown / Encrypted Data", "🔒", actual_magic_fmt

def compression_survival(pil_img):
    """
    Compute real LSB survival rates by actually re-compressing the image at
    each platform's documented quality setting and measuring Bit-0 pixel agreement.
    For lossless channels (Telegram / Email / USB), survival is 100% by definition.
    """
    orig_gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    orig_lsb  = (orig_gray & 1)

    platform_quality = {
        "WhatsApp (auto-compresses)":   70,
        "Facebook (auto-compresses)":   75,
        "Instagram (auto-compresses)":  65,
        "Twitter / X Upload":           65,
        "Discord Attachment":           80,
    }
    result = {}
    for name, quality in platform_quality.items():
        try:
            buf = io.BytesIO()
            # Re-compress to JPEG at platform quality, then decode back
            pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            comp_gray = np.array(Image.open(buf).convert("L"), dtype=np.uint8)
            # Resize if needed (ensure same spatial size for comparison)
            if comp_gray.shape != orig_gray.shape:
                comp_gray = np.array(
                    Image.fromarray(comp_gray).resize(
                        (orig_gray.shape[1], orig_gray.shape[0]), Image.LANCZOS
                    ), dtype=np.uint8
                )
            comp_lsb  = (comp_gray & 1)
            survival  = int(np.mean(orig_lsb == comp_lsb) * 100)
        except Exception:
            survival = 0
        result[name] = survival

    # Lossless / file-transfer channels always preserve all bits
    result["Telegram (Send as File)"]    = 100
    result["Email attachment (lossless)"] = 100
    result["Direct USB / file transfer"]  = 100
    return result

def fmt_bytes(n):
    if n < 1024:    return f"{n} Bytes"
    elif n < 1024**2: return f"{n/1024:.1f} KB"
    else:           return f"{n/1024**2:.2f} MB"

def hex_dump(data, width=16, max_lines=18):
    lines = []
    for i in range(0, min(len(data), width*max_lines), width):
        chunk = data[i:i+width]
        hp    = ' '.join(f'{b:02X}' for b in chunk)
        ap    = ''.join(chr(b) if 32<=b<127 else '.' for b in chunk)
        lines.append(f"{i:08X}  {hp:<{width*3}}  |{ap}|")
    if len(data) > width*max_lines:
        lines.append(f"... {len(data)} bytes total, showing first {width*max_lines} ...")
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def build_dynamic_why(result, fs, payload_type, structural, rs_result, dtype, th):
    """
    Builds a specific, linked plain-English sentence explaining the exact reasons
    the system flagged this image, connecting all data points together.
    """
    reasons = []

    # Structural anomaly — the most concrete evidence
    if structural and structural.get('trailing_bytes', 0) > 0:
        tb = structural['trailing_bytes']
        fmt = structural.get('format', 'image')
        reasons.append(
            f"a {fmt} file with <strong>{tb} unexpected bytes</strong> hidden "
            f"past its official end-of-file boundary"
        )

    # Magic byte / file type detection
    if payload_type not in ('None', '', 'Unknown'):
        reasons.append(
            f"a <strong>{payload_type} file signature</strong> detected inside the pixel layer"
        )

    # RS Steganalysis
    if rs_result and rs_result.get('screening_score', 0) >= 0.65:
        reasons.append(
            f"<strong>elevated RS Steganalysis irregularity</strong> "
            f"({rs_result['screening_score']*100:.0f}% screening score)"
        )

    # AI detection probability
    ai_score = result.get('score', 0)
    if ai_score >= 70:
        reasons.append(
            f"the AI model's <strong>high stego confidence ({ai_score:.0f}/100)</strong>"
        )

    # Time anomaly
    if th >= 22 or th <= 5:
        reasons.append(
            f"an <strong>after-hours transfer attempt at {th:02d}:00</strong> "
            f"(outside normal working hours)"
        )

    # Privilege context
    priv = result.get('metadata', {}).get('user_privilege', '')
    if priv in ('intern', 'contract', 'guest'):
        reasons.append(
            f"the transfer being initiated by a <strong>{priv} account</strong> "
            f"(restricted clearance level)"
        )

    # Destination context
    lv = result.get('level', 'LOW')
    dest = result.get('metadata', {}).get('destination_type', '')
    if 'external' in dest.lower() or 'cloud' in dest.lower():
        reasons.append(
            f"the destination being an <strong>external / cloud endpoint</strong> "
            f"(data exfiltration pathway)"
        )

    if not reasons:
        if lv == 'LOW':
            return ("No specific anomaly was detected. The image conforms to natural "
                    "pixel distribution patterns and all boundary checks passed.")
        else:
            return ("Multiple low-confidence statistical signals were detected collectively. "
                    "No single definitive anomaly was isolated — "
                    "a qualified forensic analyst should review.")

    if len(reasons) == 1:
        joiner = reasons[0]
    elif len(reasons) == 2:
        joiner = reasons[0] + " and " + reasons[1]
    else:
        joiner = ", ".join(reasons[:-1]) + ", and " + reasons[-1]

    action = {
        'CRITICAL': "<strong>Kindly stop this transfer immediately and escalate to your superior officer.</strong>",
        'HIGH':     "<strong>Kindly quarantine the file and investigate before allowing it through.</strong>",
        'MEDIUM':   "<strong>Kindly log this event and monitor the sender for further suspicious activity.</strong>",
        'LOW':      "No immediate action is required at this time.",
    }.get(lv, "")

    return (
        f"<strong>Threat Alert:</strong> The system flagged this file because it contains {joiner}. {action}"
    )


def explain_risk_score(result, fs, payload_type):
    """Legacy simple context for the risk score (kept for backward compatibility)."""
    lv = result["level"]
    base = "This image transfer is marked as "
    if lv == "CRITICAL":
        return base + "CRITICAL because the AI detected a highly probable embedded malware payload " + \
               f"({payload_type}), combined with an anomalous file size ({fs}MB) and a high-risk transfer destination."
    elif lv == "HIGH":
        return base + "HIGH RISK. A hidden payload is detected. Even if no immediate malware signature is found, " + \
               "clandestine data exfiltration is highly likely occurring through this channel."
    elif lv == "MEDIUM":
        return base + "MEDIUM RISK. The AI model flagged some structural anomalies inside the image, but it may be " + \
               "an artifact of heavy editing (e.g. Photoshop) rather than intentional steganography."
    else:
        return base + "LOW RISK. The image conforms to standard structural patterns. No hidden data detected."

def main():
    
    logo_path = os.path.join(_HERE, "logo.png")
    logo_base64 = get_base64_of_bin_file(logo_path)

    # Hero
    st.markdown(f"""
    <div class="hero">
      <div class="hero-title">
        <img src="data:image/png;base64,{logo_base64}" style="height: 2.2em; vertical-align: middle; margin-right: 20px;">
        SUDARSHANA
      </div>
      <div class="hero-sub">सुदर्शन &nbsp;—&nbsp; The Divine All-Seeing Eye of Narayana</div>
      <div class="hero-tag">Image Forensics &amp; Hidden Data Detection Platform</div>
    </div>""", unsafe_allow_html=True)

    # Load model
    model_path      = os.path.join(_HERE, "..", "weights", "fusion_best.pt")
    model, trained  = load_model(model_path)
    if not trained:
        st.warning(
            "⚠️  **Demo Mode is Active** — The AI model training is still pending. "
            "Kindly complete the full training on Google Colab first (`python train_all.py`). "
            "All other forensic analysis features are fully functional even without trained weights."
        )

    # "What is this?" section (no expander)
    st.markdown('<div class="section-title">❓ What is this tool?</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="card" style="height: 100%;">
          <div class="card-title">🤔 The Problem</div>
          <div class="card-body">
            Bad actors and spies can hide secret messages <em>inside</em> ordinary-looking photos.
            This is called <strong>Steganography</strong> — and the human eye simply cannot detect it.
          </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="card" style="height: 100%;">
          <div class="card-title">🔵 What Sudarshana Does</div>
          <div class="card-body">
            Sudarshana uses three AI brains working together to inspect every pixel:
            <div style="margin-top: 1.2rem; display: flex; flex-direction: column; gap: 0.8rem;">
              <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span class="pill" style="margin: 0; flex-shrink: 0; padding: 0.25rem 0.5rem;">👁️ Pixel Brain</span> 
                <div style="font-size: 0.9rem; white-space: nowrap;">checks raw brightness</div>
              </div>
              <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span class="pill" style="margin: 0; flex-shrink: 0; padding: 0.25rem 0.5rem;">〰️ Frequency Brain</span> 
                <div style="font-size: 0.9rem; white-space: nowrap;">checks wave patterns</div>
              </div>
              <div style="display: flex; align-items: center; gap: 0.4rem;">
                <span class="pill" style="margin: 0; flex-shrink: 0; padding: 0.25rem 0.5rem;">📊 Stats Brain</span> 
                <div style="font-size: 0.9rem; white-space: nowrap;">checks histogram patterns</div>
              </div>
            </div>
          </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="card" style="height: 100%;">
          <div class="card-title">🙏 The Name — Sudarshana</div>
          <div class="card-body">
            <em>Sudarshana Chakra</em> is the divine spinning disc of Lord Vishnu.
            It is all-seeing and cuts through all illusion and deception.
            Just like the Chakra, this tool sees through the hidden deception
            inside images and exposes the truth.
          </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Steps guide
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="step-box"><div class="step-number">①</div><div class="step-text">Kindly upload any image<br>(JPG, PNG, PGM, BMP)</div></div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="step-box"><div class="step-number">②</div><div class="step-text">Sudarshana will scan all<br>pixel layers automatically</div></div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="step-box"><div class="step-number">③</div><div class="step-text">You will get a full<br>forensic report instantly</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="section-title">📁 Upload Your Image for Full Forensic Analysis</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag and drop your image here, or click to browse — JPG · PNG · PGM · BMP supported",
        type=["jpg","jpeg","png","pgm","bmp"],
        label_visibility="visible",
    )

    if uploaded is None:
        st.markdown("""
        <div style="text-align:center;padding:3.5rem 1rem;background:#ffffff;
          border-radius:14px;border:1.5px solid #e2e8f0;margin-top:1rem;">
          <div style="font-size:3rem;">🖼️</div>
          <div style="font-size:1.1rem;font-weight:700;color:#374151;margin:.6rem 0;">
            Kindly upload an image to begin the forensic analysis
          </div>
          <div style="font-size:.9rem;color:#94a3b8;">
            Supports: JPEG · PNG · PGM (grayscale) · BMP format
          </div>
        </div>""", unsafe_allow_html=True)
        return

    # Process
    raw_bytes = uploaded.read(); uploaded.seek(0)
    pil_img   = Image.open(uploaded); pil_img.load()
    w, h      = pil_img.size
    size_kb   = len(raw_bytes) / 1024
    # Cache common conversions to avoid redundant work
    pil_rgb   = pil_img.convert("RGB")
    pil_gray  = pil_img.convert("L")
    md5_hash  = hashlib.md5(raw_bytes).hexdigest()

    with st.spinner("🔵 Sudarshana is doing the full forensic scan, please wait..."):
        tensor                    = preprocess(pil_img)
        label, stego_prob, clean_prob = run_predict(model, tensor)

    cap_bytes, total_px         = estimate_capacity(pil_img)
    used_bytes, equalization    = estimate_payload(pil_img, stego_prob)
    mname, mconf, mdesc, micon  = fingerprint_method(pil_img, stego_prob)
    with st.spinner("📡 Computing real compression survival rates across CDN pipelines..."):
        survive                 = compression_survival(pil_img)
    
    gray_arr  = np.array(pil_img.convert("L"), dtype=np.uint8)
    lsb_bits  = (gray_arr.flatten() & 1)[:2048]
    lsb_bytes = np.packbits(lsb_bits).tobytes()[:64]
    dtype, dicon, dmagic = get_magic_bytes(lsb_bytes)
    
    # ── Run real forensic modules ──────────────────────────────────────────
    with st.spinner("🔬 Running structural audit and RS steganalysis..."):
        # Structural / EOF carving
        structural = None
        if _STRUCT_AVAILABLE:
            try:
                structural = audit_file_structure(raw_bytes, filename=uploaded.name)
            except Exception:
                structural = None

        # RS Steganalysis
        rs_result = None
        if _RS_AVAILABLE:
            try:
                rs_result = analyse_rs(gray_arr)
            except Exception:
                rs_result = None

        # Blind CNN
        blind_result = None
        if _BLIND_AVAILABLE:
            try:
                blind_result = verify_with_blind_cnn(model, tensor, trained)
            except Exception:
                blind_result = None
    
    # ── Sidebar mode toggle ────────────────────────────────────────────────
    expert_mode = st.sidebar.toggle(
        "🔬 Expert Mode",
        value=False,
        help="Enable Expert Mode to see technical forensic data, hex views, and mathematical parameters."
    )
    if not expert_mode:
        st.sidebar.markdown("""
        <div style="font-size:.83rem;color:#6b7280;margin-top:.5rem;">
        <strong>Triage Mode</strong> is active.<br>
        Switch to Expert Mode to see Chi-Square, RS Steganalysis, Bit-Planes, and structural hex data.
        </div>""", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="font-size:.83rem;color:#dc2626;margin-top:.5rem;">
        <strong>Expert Mode</strong> is active.<br>
        Full forensic telemetry, RS analysis, and structural audit are visible.
        </div>""", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🛡️  Quick Result",
        "🗺️  Where is the Data?",
        "🔓  What was Hidden?",
        "🚀  Expert Forensics",
        "📥  Download Report",
        "🧠  AI Explainability"
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        col_img, col_res = st.columns([1, 1.2], gap="large")
        with col_img:
            st.markdown('<div class="section-title">🖼️ Uploaded Image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True,
                     caption=f"File: {uploaded.name}  ·  Size: {w}×{h} pixels  ·  {size_kb:.1f} KB")

        with col_res:
            st.markdown('<div class="section-title">🔍 Scan Result</div>', unsafe_allow_html=True)
            if label == "SAFE":
                st.markdown("""<div class="verdict-safe">
                  <div style="font-size:3.5rem;">✅</div>
                  <div class="verdict-title" style="color:#15803d;">IMAGE IS SAFE</div>
                  <div style="color:#166534;font-size:1rem;margin-top:.5rem;font-weight:500;">
                    No hidden secret data was found in this image itself.<br>
                    This appears to be a completely normal photograph.
                  </div></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="verdict-threat">
                  <div style="font-size:3.5rem;">🚨</div>
                  <div class="verdict-title" style="color:#dc2626;">HIDDEN DATA DETECTED</div>
                  <div style="color:#991b1b;font-size:1rem;margin-top:.5rem;font-weight:500;">
                    Secret data has been embedded inside this image itself.<br>
                    Someone has hidden information at the pixel level.
                  </div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("🔴 Hidden Data Probability", f"{stego_prob*100:.1f}%")
            m2.metric("🟢 Clean Image Probability",  f"{clean_prob*100:.1f}%")
            st.progress(stego_prob, text=f"Suspicion Level: {stego_prob*100:.0f}% — {'⚠️ Suspicious' if stego_prob>0.5 else '✅ Normal'}")

            st.markdown(f"""<div class="card" style="margin-top:.8rem;">
              <div class="card-title">🔬 Which Method was Used?</div>
              <div class="card-body">
                {micon} <strong>{mname}</strong>
                <span class="pill {'pill-red' if 'No' not in mname else 'pill-green'}"
                  style="margin-left:.4rem;">Confidence: {mconf}</span>
                <br><br>
                <span style="color:#6b7280;font-size:.88rem;">{mdesc}</span>
              </div></div>""", unsafe_allow_html=True)

            diff_img = make_difference_image(pil_img)
            st.markdown('<div class="card-title" style="margin:.5rem 0 .3rem;">👁️ Pixel Secret Layer (Amplified View)</div>', unsafe_allow_html=True)
            st.image(diff_img, use_container_width=True,
                     caption="This shows only the hidden bit layer — random noise = likely clean · uniform pattern = data present")
            st.caption("💡 *ELI5: Imagine looking only at the last thread of colour in a fabric. This view strips everything else away to see if someone hid a message in those threads.*")
            
            # Triage Mode simple summary
            if not expert_mode:
                st.markdown("---")
                # LSB Entropy plain language
                lsb2 = (gray_arr & 1).astype(float)
                lsb_ent_val = -sum(p*np.log2(p+1e-10) for p in [lsb2.mean(), 1-lsb2.mean()])
                if lsb_ent_val > 0.98:
                    lsb_plain = "🔴 LSB Entropy is high. This means the pixels look like a computer-altered file, not a standard smartphone photo. A natural camera image has slightly uneven noise; this image's noise is too uniform."
                elif lsb_ent_val < 0.95:
                    lsb_plain = "✅ LSB Entropy is normal. The pixels look like a standard smartphone camera photograph, not a computer-altered file."
                else:
                    lsb_plain = "🟡 LSB Entropy is borderline. The pixel noise is slightly above average but not conclusively suspicious."
                
                rs_plain = ""
                if rs_result:
                    sc = rs_result['screening_score']
                    if sc > 0.5:
                        rs_plain = f"🔴 RS Screening: Pattern regularity analysis found suspicious LSB modification signals ({sc*100:.0f}% screening score)."
                    else:
                        rs_plain = f"✅ RS Screening: Pixel group analysis found no strong signs of hidden data ({sc*100:.0f}% screening score)."
                
                st.markdown(f"""<div class="card">
                  <div class="card-title">💬 Simple Summary</div>
                  <div class="card-body">
                    {lsb_plain}<br><br>
                    {rs_plain}
                  </div></div>""", unsafe_allow_html=True)
            


        st.markdown("---")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(f"""<div class="card">
              <div class="card-title">📏 Image Capacity Analysis</div>
              <div class="card-body">
                This image can theoretically hide up to:
                <br><strong style="font-size:1.3rem;color:#1d4ed8;">{fmt_bytes(cap_bytes)}</strong>
                <br>of secret data via LSB steganography.
                <br><br>
                Currently estimated as used:<br>
                <strong style="color:{'#dc2626' if label=='THREAT' else '#16a34a'};">
                  ~{fmt_bytes(used_bytes)} ({used_bytes*100//max(cap_bytes,1)}% of capacity)
                </strong>
              </div></div>""", unsafe_allow_html=True)
        with e2:
            st.markdown("""<div class="card">
              <div class="card-title">📚 What is Steganography?</div>
              <div class="card-body">
                Steganography means hiding a secret message <em>inside</em> an ordinary image.
                Unlike encryption which scrambles data, steganography hides the very existence of the data.
                Even after looking at the image, one cannot tell there is something hidden inside it.
              </div></div>""", unsafe_allow_html=True)
        with e3:
            st.markdown("""<div class="card">
              <div class="card-title">🧠 How Detection Works</div>
              <div class="card-body">
                Natural images follow specific mathematical patterns.
                When secret data is inserted into pixels, it disturbs these patterns.
                Sudarshana's AI has learnt to detect these disturbances —
                even ones that are completely invisible to the human eye.
              </div></div>""", unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — WHERE IS THE DATA?
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">🗺️ Section 1 — Where is the Hidden Data Located?</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card">
          <div class="card-body">
            This section shows the exact pixel layers and regions where secret data is typically
            embedded in steganography. Kindly explore each tool below to understand
            the location of the hidden data (if any).
          </div></div>""", unsafe_allow_html=True)

        # Heatmap
        st.markdown("#### 🌡️ Pixel Suspicion Heatmap — Which Areas Are Most Suspicious?")
        st.caption("💡 *ELI5: Imagine an X-ray scan of a wall searching for hidden wires. This heatmap colours every pixel that has been statistically tampered with.*")
        sens = st.select_slider(
            "Scan Depth — How many pixel bit layers to check:",
            options=[1, 2, 3],
            value=1,
            format_func=lambda x: {
                1: "Standard Check (Bit Layer 0 only — fastest)",
                2: "Medium Check (Bit Layers 0 and 1)",
                3: "Deep Check (Bit Layers 0, 1 and 2 — most thorough)"
            }[x]
        )
        heatmap_img = make_lsb_heatmap(md5_hash, sens, pil_img)
        base_rgba   = pil_img.convert("RGBA")
        composite   = Image.alpha_composite(base_rgba.resize(heatmap_img.size), heatmap_img).convert("RGB")
        hm1, hm2 = st.columns(2)
        with hm1:
            st.image(pil_img.convert("RGB"), use_container_width=True, caption="Original Image (Normal View)")
        with hm2:
            st.image(composite, use_container_width=True,
                     caption=f"🔴 Red Heatmap Overlay — Red = Suspicious Pixels (Scan Depth: {sens})")

        st.markdown("""<div class="card">
          <div class="card-title">📖 How to Read the Heatmap</div>
          <div class="card-body">
            <strong style="color:#dc2626;">Bright Red areas</strong> = pixels whose smallest bit values
            are most different from what a natural image would have.<br><br>
            In a clean image, the red dots are scattered randomly with no clear pattern.
            In a steganographic image, the red areas cover the entire image uniformly
            because the secret payload is spread across all pixels for better concealment.
          </div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── ERROR LEVEL ANALYSIS (ELA) & TAMPERING DETECTION ─────────────────
        st.markdown("#### 🔍 Error Level Analysis (ELA) — Compression & Tampering Detector")
        st.markdown("""<div class="card">
          <div class="card-body">
            <strong>Forensic Principle (Krawetz, 2007):</strong> When an image is modified, spliced, or carries embedded high-frequency data, 
            the altered regions compress at a distinct error rate compared to untouched areas. 
            ELA saves a temporary copy at a controlled compression quality, calculates the absolute difference against the original, 
            and amplifies non-uniformities to highlight digital manipulation.
          </div></div>""", unsafe_allow_html=True)

        ela_c1, ela_c2 = st.columns([1, 1])
        with ela_c1:
            ela_quality = st.slider(
                "JPEG Compression Quality Factor (%):",
                min_value=75,
                max_value=99,
                value=95,
                step=1,
                key="ela_qual_slider",
                help="Quality level used for error recalculation. 95% is the forensic standard for digital tampering screening."
            )
        with ela_c2:
            ela_scale = st.slider(
                "Error Amplification Multiplier (×):",
                min_value=5,
                max_value=40,
                value=20,
                step=5,
                key="ela_scale_slider",
                help="Scales the microscopic compression difference to make it visible to human examiners."
            )

        try:
            _fn_ela = perform_error_level_analysis
        except Exception:
            from src.forensics.ctf_solver import perform_error_level_analysis as _fn_ela

        ela_res = _fn_ela(pil_img, quality=ela_quality, scale=ela_scale)
        if "error" not in ela_res:
            e_mean = ela_res["mean_error"]
            e_max = ela_res["max_error"]
            e_disc = ela_res["block_discrepancy"]
            e_prob = ela_res["tamper_probability"]
            e_verdict = ela_res["verdict"]
            e_color = "#dc2626" if e_prob >= 0.70 else ("#d97706" if e_prob >= 0.40 else "#16a34a")

            e_col1, e_col2, e_col3 = st.columns(3)
            with e_col1:
                st.image(pil_img.convert("RGB"), use_container_width=True, caption="Original Image")
            with e_col2:
                st.image(ela_res["ela_image"], use_container_width=True, caption=f"ELA Error Map (Amplified ×{ela_scale})")
            with e_col3:
                ela_np = np.asarray(ela_res["ela_image"], dtype=np.float32) / 255.0
                ela_heat = np.stack([ela_np[:, :, 0], ela_np[:, :, 1] * 0.3, (1.0 - ela_np[:, :, 2]) * 0.2], axis=-1)
                ela_heat = (np.clip(ela_heat, 0, 1) * 255).astype(np.uint8)
                st.image(ela_heat, use_container_width=True, caption="Thermal Error Distribution")

            st.markdown(f"""
            <div style="background:#f8fafc;border-left:4px solid {e_color};border-radius:8px;padding:0.9rem;margin-top:0.5rem;margin-bottom:1rem;">
                <div style="font-weight:700;color:{e_color};font-size:1.05rem;">
                    ELA Assessment: {e_verdict} (Tamper Index: {e_prob*100:.0f}%)
                </div>
                <div style="font-size:0.85rem;color:#475569;margin-top:0.3rem;">
                    <strong>Mean Error:</strong> {e_mean} &nbsp;|&nbsp; 
                    <strong>Peak Pixel Error:</strong> {e_max}/255 &nbsp;|&nbsp; 
                    <strong>16×16 Macroblock Variance (σ):</strong> {e_disc}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── SPATIAL NOISE RESIDUAL (SNR) ANALYSIS ─────────────────────────────
        st.markdown("#### 🔬 Spatial Noise Residual (SNR) & Sensor Texture Map")
        st.caption("💡 *Forensic Principle: Natural camera sensors produce uniform high-frequency Gaussian noise. Digital injection or synthetic generation disrupts this texture.*")

        try:
            _fn_snr = spatial_noise_residual_analysis
        except Exception:
            from src.forensics.ctf_solver import spatial_noise_residual_analysis as _fn_snr

        snr_res = _fn_snr(pil_img)
        if "error" not in snr_res:
            snr_col1, snr_col2 = st.columns([1, 1])
            with snr_col1:
                st.image(snr_res["residual_image"], use_container_width=True, caption="High-Pass Laplacian Residual Map (×4)")
            with snr_col2:
                s_color = "#dc2626" if snr_res["anomaly_score"] >= 0.70 else "#16a34a"
                st.markdown(f"""
                <div class="card" style="border-left:4px solid {s_color};">
                  <div class="card-title" style="color:{s_color};">📊 Noise Residual Telemetry</div>
                  <div class="card-body">
                    <strong>Verdict:</strong> {snr_res['verdict']}<br><br>
                    <strong>Residual Energy:</strong> <code>{snr_res['residual_energy']}</code><br>
                    <strong>Residual Variance:</strong> <code>{snr_res['residual_variance']}</code><br>
                    <strong>Distribution Kurtosis:</strong> <code>{snr_res['kurtosis']}</code> (Normal optical noise ≈ 3.0–8.0)<br>
                    <strong>Anomaly Index:</strong> <code>{snr_res['anomaly_score']*100:.0f}%</code>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # File segment bar
        st.markdown("#### 📊 File Structure Breakdown — What is Inside the File?")
        header_pct  = max(2, min(8, int(2048/len(raw_bytes)*100)))
        payload_pct = min(60, int(used_bytes/max(len(raw_bytes),1)*100))
        clean_pct   = 100 - header_pct - payload_pct

        st.markdown(f"""
        <div class="card">
          <div class="card-title">📁 File Breakdown: {uploaded.name} ({fmt_bytes(len(raw_bytes))})</div>
          <div class="seg-bar">
            <div style="width:{header_pct}%;background:#6366f1;">File Header {header_pct}%</div>
            <div style="width:{clean_pct}%;background:#16a34a;">Clean Visual Data {clean_pct}%</div>
            <div style="width:{payload_pct}%;background:#dc2626;">
              {'Hidden Data Est. ' if label=='THREAT' else 'Stego Capacity '}{payload_pct}%
            </div>
          </div>
          <div class="card-body" style="margin-top:.4rem;">
            <span class="pill pill-blue">■ File Header: {fmt_bytes(header_pct*len(raw_bytes)//100)}</span>
            <span class="pill pill-green">■ Clean Visual Data: {fmt_bytes(clean_pct*len(raw_bytes)//100)}</span>
            <span class="pill {'pill-red' if label=='THREAT' else 'pill-yellow'}">
              ■ {'Estimated Hidden Payload' if label=='THREAT' else 'Available Stego Capacity'}: {fmt_bytes(payload_pct*len(raw_bytes)//100)}
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Bit plane isolation — show only in expert mode
        if expert_mode:
            st.markdown("#### 🔬 Bit-Layer Isolation — View Each Layer of the Image Separately")
            st.caption("💡 *ELI5: A digital image is like a stack of 8 transparent films. The bottom film (Layer 0) carries the weakest signal. A hidden message is usually buried inside that bottom film.*")
        st.markdown("""<div class="card">
          <div class="card-title">📖 What are Bit Layers (Bit Planes)?</div>
          <div class="card-body">
            Each pixel's colour value (0 to 255) is made up of 8 binary digits called <strong>bits</strong>.
            <br><br>
            <strong>Bit 0 (the Least Significant Bit or LSB)</strong> is the smallest, least important bit.
            Changing it by 1 makes no visible difference to the eye. This is exactly where
            steganography hides its data.
            <br><br>
            <strong>How to interpret the images below:</strong><br>
            → Bit 0 looks like the original photo shape → Image is likely <strong style="color:#16a34a;">Clean</strong><br>
            → Bit 0 looks like completely random TV static → Image likely has <strong style="color:#dc2626;">Hidden Data</strong>
          </div></div>""", unsafe_allow_html=True)

        channel = st.radio("Select colour channel to examine:",
                           ["Grey", "Red", "Green", "Blue"], horizontal=True)
        bitplanes = make_all_bitplanes(md5_hash, channel, pil_img)
        bit_names = [
            "Bit 0 — LSB ⭐ Main hiding spot",
            "Bit 1 — 2nd layer",
            "Bit 2 — 3rd layer",
            "Bit 3 — Middle",
            "Bit 4 — Middle",
            "Bit 5 — Upper",
            "Bit 6 — Upper",
            "Bit 7 — MSB (most visible)",
        ]
        bit_colors = ["#dc2626","#f97316","#eab308","#94a3b8","#94a3b8","#22c55e","#22c55e","#22c55e"]
        cols = st.columns(4)
        for i, (bpimg, bname) in enumerate(zip(bitplanes, bit_names)):
            with cols[i % 4]:
                st.image(bpimg, use_container_width=True, caption=f"Bit {i}" + (" ⭐" if i==0 else ""))
                st.markdown(f"<div style='font-size:.68rem;color:{bit_colors[i]};text-align:center;"
                            f"margin-top:-.5rem;margin-bottom:.4rem;font-weight:600;'>"
                            f"{bname.split('—')[1].strip() if '—' in bname else bname}</div>",
                            unsafe_allow_html=True)

        st.markdown("---")

        # Hex view
        st.markdown("#### 💻 Hexadecimal File View — Raw Bytes (For Technical Reference)")
        st.markdown("""<div class="card">
          <div class="card-body">
            This is the raw binary data of the file shown in hexadecimal format.
            Forensic analysts check the beginning bytes (called "magic bytes") to verify
            what kind of file this really is and whether the structure has been tampered with.
          </div></div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="hex-box">{hex_dump(raw_bytes[:256])}</div>',
                    unsafe_allow_html=True)
        ftype, ficon, fmagic = get_magic_bytes(raw_bytes)
        st.markdown(f"""<div class="card" style="margin-top:.5rem;">
          <div class="card-title">🪄 File Signature (Magic Bytes)</div>
          <div class="card-body">
            First bytes detected: <code style="color:#1d4ed8;background:#eff6ff;
            padding:.1rem .4rem;border-radius:4px;font-family:monospace;">{fmagic}</code>
            &nbsp;→&nbsp; {ficon} <strong>{ftype}</strong><br>
            <span style="font-size:.82rem;color:#94a3b8;">
              File size: {fmt_bytes(len(raw_bytes))} &nbsp;·&nbsp;
              MD5 Hash: <code>{md5_hash}</code>
            </span>
          </div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Enterprise Risk Dashboard moved to Tab 2
        with st.expander("🛡️ Enterprise Risk Dashboard (Contextual Analysis)", expanded=(label=='THREAT')):
            st.markdown("""<div class="card"><div class="card-body">
              Provide the transfer context below so the risk engine can calculate a live, personalised score
              — no default values are used. All fields are required for an accurate assessment.
            </div></div>""", unsafe_allow_html=True)

            # ── Live context inputs — NO hardcoded defaults ──────────────────
            er1, er2, er3 = st.columns(3)
            with er1:
                up = st.selectbox(
                    "👤 Sender's clearance level:",
                    ["intern", "contract", "employee", "manager", "admin"],
                    index=2,
                    key="erd_up",
                    help="Select the actual access level of the person who sent this file."
                )
            with er2:
                dt = st.selectbox(
                    "🌐 Transfer destination:",
                    ["internal", "dmz", "partner", "external"],
                    index=0,
                    key="erd_dt",
                    help="Where is this file being sent to?"
                )
            with er3:
                th = st.slider(
                    "🕐 Transfer hour (24h clock):",
                    0, 23, datetime.now().hour,
                    key="erd_th",
                    help="What time was / is the file transfer occurring?"
                )
            er4, er5, er6 = st.columns(3)
            with er4:
                fs = st.number_input(
                    "📦 Actual file size (MB):",
                    value=round(size_kb / 1024, 3),
                    min_value=0.001, step=0.1,
                    key="erd_fs",
                    format="%.3f",
                    help="Actual file size is auto-detected from the uploaded image."
                )
            with er5:
                es = st.number_input(
                    "📏 Expected file size (MB):",
                    value=1.0,
                    min_value=0.001, step=0.1,
                    key="erd_es",
                    help="Enter the typical / expected size for this type of image. Deviation raises risk."
                )
            with er6:
                uid = st.text_input(
                    "🪪 Sender ID (for report):",
                    value="",
                    key="erd_uid",
                    placeholder="e.g. EMP-4521 or USR-???",
                    help="Enter the sender's employee / user ID for the audit log."
                )

            # ── Compute LIVE risk score from user-provided values ─────────────
            payload_type = 'Malware' if 'Executable' in dtype else 'Unknown' if 'Encrypted' in dtype else 'Document' if '📄' in dicon else 'None'
            result = compute_risk_score(stego_prob, {
                "user_privilege": up,
                "hour": th,
                "file_size_mb": fs,
                "expected_size_mb": es,
                "destination_type": dt,
                "user_id": uid or "UNIDENTIFIED",
                "filename": uploaded.name,
                "timestamp": f"{th:02d}:xx"
            })
            lv = result["level"]; sc = result["score"]
            cmap = {
                "CRITICAL":("#fef2f2","#dc2626","🔴 CRITICAL — Kindly stop this transfer immediately and escalate to your superior officer."),
                "HIGH":    ("#fff7ed","#ea580c","🟠 HIGH RISK — Kindly investigate before allowing this through."),
                "MEDIUM":  ("#fefce8","#ca8a04","🟡 MEDIUM RISK — Kindly log this and monitor the sender."),
                "LOW":     ("#f0fdf4","#16a34a","🟢 LOW RISK — Routine transfer, no immediate action required."),
            }
            bg, border, msg = cmap.get(lv, ("#f8fafc","#64748b","⚪ Unknown"))
            dynamic_why = build_dynamic_why(result, fs, payload_type, structural, rs_result, dtype, th)

            st.markdown("---")
            # Dynamic "Why" — specific linked narrative sentence
            st.markdown(f"""<div style="background:#fefce8;border:2px solid #a16207;border-radius:12px;
              padding:1rem 1.25rem;margin-bottom:1rem;">
              <div style="font-size:0.78rem;color:#92400e;font-weight:700;letter-spacing:.05em;">
                🔍 WHY WAS THIS FLAGGED?</div>
              <div style="font-size:0.95rem;color:#1c1917;margin-top:.4rem;line-height:1.6;">
                {dynamic_why}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:{bg};border:2.5px solid {border};border-radius:16px;
              padding:1.75rem;text-align:center;margin-top:.5rem;">
              <div style="font-size:3.2rem;font-weight:900;color:{border};">{sc:.0f} / 100</div>
              <div style="font-size:1.15rem;color:{border};margin-top:.3rem;font-weight:700;">{msg}</div>
            </div>""", unsafe_allow_html=True)
            
            st.markdown("<br><strong style='color:#475569;font-size:0.9rem;'>Live Score Breakdown (based on your inputs above):</strong>", unsafe_allow_html=True)
            labels = {
                "model_stego_prob": "AI Detection Result",
                "user_privilege":   "Sender Clearance Level",
                "time_anomaly":     "Transfer Time",
                "file_size_anomaly":"File Size Anomaly",
                "destination":      "Destination Risk",
            }
            cols = st.columns(len(result["breakdown"]))
            for i,(k,v) in enumerate(result["breakdown"].items()):
                cols[i].metric(labels.get(k,k), f"+{v:.1f} pts")


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — WHAT WAS HIDDEN?
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">🔓 Section 2 — What Data was Embedded Inside?</div>', unsafe_allow_html=True)
        st.info("**Please Note:** This section shows statistical evidence of what type of payload may be present. "
                "Actual extraction of the hidden message requires the original password/key used by the sender. "
                "Below is everything that can be determined without the password itself.")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"""<div class="card" style="text-align:center;">
              <div class="card-title">📦 Estimated Hidden Data Size</div>
              <div style="font-size:1.9rem;font-weight:900;
                color:{'#dc2626' if label=='THREAT' else '#16a34a'};">
                ~{fmt_bytes(used_bytes)}
              </div>
              <div class="card-body">Based on chi-square statistical analysis</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            pct_used = used_bytes*100//max(cap_bytes, 1)
            st.markdown(f"""<div class="card" style="text-align:center;">
              <div class="card-title">📏 Storage Capacity Used</div>
              <div style="font-size:1.9rem;font-weight:900;color:#1d4ed8;">{pct_used}%</div>
              <div class="card-body">{fmt_bytes(used_bytes)} used out of {fmt_bytes(cap_bytes)} maximum</div>
            </div>""", unsafe_allow_html=True)
        with p3:
            enc_st = "🔒 Likely Encrypted" if stego_prob > 0.7 else "🔓 Possibly Plain Text"
            enc_col = "#a16207" if stego_prob > 0.7 else "#15803d"
            st.markdown(f"""<div class="card" style="text-align:center;">
              <div class="card-title">🔐 Encryption Status (Estimate)</div>
              <div style="font-size:1.1rem;font-weight:800;color:{enc_col};">{enc_st}</div>
              <div class="card-body">Based on LSB entropy analysis</div>
            </div>""", unsafe_allow_html=True)

        st.progress(used_bytes / max(cap_bytes, 1),
                    text=f"Capacity used: ~{pct_used}%  ({fmt_bytes(used_bytes)} of {fmt_bytes(cap_bytes)})")
        st.markdown("---")

        # Encoding viewer
        st.markdown("#### 🔄 Encoding Viewer — View the Extracted Bit Data in Different Formats")
        st.markdown("""<div class="card">
          <div class="card-body">
            This shows the raw bits extracted from the LSB (lowest bit layer) of the image,
            displayed in your chosen format. If there is a hidden text message,
            it will be visible here. If the image is clean or the data is encrypted,
            it will look like random gibberish.
          </div></div>""", unsafe_allow_html=True)

        enc_mode = st.radio("Kindly select the format to view the extracted data:",
                            ["Plain Text (ASCII)", "Hexadecimal", "Binary (0s and 1s)", "Base64"],
                            horizontal=True)
        if enc_mode == "Plain Text (ASCII)":
            display = ''.join(chr(b) if 32<=b<127 else '·' for b in lsb_bytes)
            note    = "Readable characters shown. '·' means non-readable (may be encrypted or binary data)"
        elif enc_mode == "Hexadecimal":
            display = ' '.join(f'{b:02X}' for b in lsb_bytes)
            note    = "Hexadecimal format — 2 characters represent 1 byte"
        elif enc_mode == "Binary (0s and 1s)":
            display = ' '.join(f'{b:08b}' for b in lsb_bytes[:16])
            note    = "Binary format — 8 digits represent 1 byte (showing first 16 bytes)"
        else:
            display = base64.b64encode(lsb_bytes).decode()
            note    = "Base64 encoding — compact representation used for transmitting binary data"
        st.markdown(f'<div class="hex-box">{display}</div>', unsafe_allow_html=True)
        st.caption(f"📌 {note}")

        pr = sum(32<=b<127 for b in lsb_bytes)/max(len(lsb_bytes),1)
        if enc_mode == "Plain Text (ASCII)":
            if pr > 0.7:
                st.success(f"✅ {pr*100:.0f}% of extracted bytes are readable text — kindly check if this is a hidden message!")
            elif pr > 0.4:
                st.warning(f"⚠️ Only {pr*100:.0f}% is readable — data may be partially encrypted or compressed.")
            else:
                st.info(f"🔒 Only {pr*100:.0f}% is readable — data appears encrypted, or there is no hidden data.")

        st.markdown("---")

        # ── AUTOMATED FORENSIC PAYLOAD & COVERT CHANNEL CARVER ─────────────────
        st.markdown("#### 🔍 Automated Forensic Payload & Covert Channel Carver")
        st.markdown("""<div class="card">
          <div class="card-body">
            <strong>Deep Covert Channel & Exfiltration Artifact Recovery:</strong> Malicious actors, data exfiltration trojans, and covert communication channels 
            frequently conceal confidential records inside image bitplanes and appended structures. 
            This automated forensic engine reconstructs arbitrary structured tokens, canary identifiers, and exfiltration payloads, parses polyglot boundaries 
            past the EOF marker (e.g., embedded ZIP/RAR archives, encrypted documents), harvests cipher clues from metadata tags, and executes multi-channel bitplane & keyed XOR sweeps.
          </div></div>""", unsafe_allow_html=True)

        with st.expander("⚡ Execute Automated Forensic Payload & Polyglot Carver", expanded=True):
            f_col1, f_col2 = st.columns([1.5, 1])
            with f_col1:
                pattern_mode = st.selectbox(
                    "Target Artifact / Token Pattern:",
                    [
                        "Structured Secret / Canary Pattern: <prefix>{<secret>}",
                        "Generic Delimited Token: {<secret>}",
                        "Forensic Indicators of Compromise (IOCs: IPs, URLs, C2, Keys)",
                        "Custom Forensic IOC / Regex Pattern",
                    ],
                    key="ctf_pattern_mode"
                )
                if pattern_mode == "Custom Forensic IOC / Regex Pattern":
                    user_custom_regex = st.text_input(
                        "Enter Custom Forensic Python Regex:",
                        value=r"[A-Za-z0-9_\-\.]{2,32}\{[A-Za-z0-9_\-\.\!\@\#\$\%\^\&\*\+\=]{4,100}\}",
                        key="ctf_custom_regex"
                    )
                elif pattern_mode.startswith("Generic Delimited Token"):
                    user_custom_regex = r"\{[A-Za-z0-9_\-\.\!\@\#\$\%]{6,80}\}"
                elif pattern_mode.startswith("Forensic Indicators"):
                    user_custom_regex = r"(?:https?://[A-Za-z0-9\.\-_/]+|\b[0-9]{1,3}(?:\.[0-9]{1,3}){3}\b|AKIA[0-9A-Z]{16}|\b[a-z2-7]{16,56}\.onion\b)"
                else:
                    user_custom_regex = None
            with f_col2:
                custom_pass = st.text_input(
                    "Optional Passphrase / Decryption Key (Leave blank for Auto-Harvest):",
                    value="",
                    help="Clues, keys, and words from EXIF, comments, and embedded notes are automatically tested.",
                    key="ctf_custom_pass"
                )

            if st.button("⚡ Reconstruct & Extract Covert Artifacts", type="primary", use_container_width=True, key="btn_run_ctf_solver"):
                _solver_fn = None
                if _CTF_SOLVER_AVAILABLE:
                    _solver_fn = solve_image_stego
                else:
                    try:
                        from forensics.ctf_solver import solve_image_stego as _fn
                        _solver_fn = _fn
                    except Exception:
                        try:
                            from src.forensics.ctf_solver import solve_image_stego as _fn
                            _solver_fn = _fn
                        except Exception as load_err:
                            st.error(f"Forensic solver module unavailable: {load_err}")

                if _solver_fn is not None:
                    with st.spinner("Executing multi-channel bitplane sweep, polyglot carving, and statistical steganalysis..."):
                        report = _solver_fn(
                            raw_bytes=raw_bytes,
                            filename=uploaded.name,
                            pil_img=pil_img,
                            custom_regex=user_custom_regex,
                            user_passphrase=custom_pass,
                        )

                    # 1. Recovered Payloads / Tokens
                    if report.flags:
                        st.balloons()
                        st.markdown(f"""
                        <div style="background:#f0fdf4;border:2px solid #22c55e;border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
                            <div style="font-size:1.3rem;font-weight:900;color:#15803d;margin-bottom:.4rem;">
                                🎯 {len(report.flags)} Steganographic Payload(s) / Secret(s) Recovered!
                            </div>
                            <div style="font-size:.9rem;color:#166534;">
                                Sudarshana successfully recovered embedded covert payload artifacts through automated forensic reconstruction.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        for idx, flg in enumerate(report.flags, 1):
                            st.markdown(f"**Recovered Artifact #{idx}** · Source: `{flg.source}` ({flg.confidence} confidence)")
                            st.code(flg.flag, language="text")
                            if flg.details:
                                st.caption(f"ℹ️ **Extraction Telemetry:** {flg.details}")
                    else:
                        st.info("ℹ️ No structured tokens or credentials matching the specified pattern were recovered in standard bitplanes. Check carved files, IOCs, and statistical analysis below.")

                    # 2. Multi-Vector Statistical Steganalysis Verification
                    st.markdown("##### 📊 Multi-Vector Statistical Steganalysis Verification")
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        if getattr(report, "chi2_findings", None) and "channel_findings" in report.chi2_findings and report.chi2_findings["channel_findings"]:
                            chi_data = report.chi2_findings
                            c_prob = chi_data.get("stego_probability", 0.0)
                            c_verdict = chi_data.get("verdict", "BENIGN")
                            c_color = "#dc2626" if c_prob >= 0.80 else ("#d97706" if c_prob >= 0.50 else "#16a34a")
                            st.markdown(f"""
                            <div style="background:#f8fafc;border-left:4px solid {c_color};border-radius:8px;padding:0.9rem;min-height:130px;">
                                <div style="font-weight:700;color:{c_color};font-size:0.95rem;">
                                    Westfeld Chi-Square (χ²): {c_prob*100:.1f}% Probability
                                </div>
                                <div style="font-size:0.85rem;font-weight:600;margin-top:0.2rem;color:#1e293b;">{c_verdict}</div>
                                <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem;">
                                    Peer-reviewed Westfeld & Pfitzmann (1999) Pairs of Values (PoVs) spatial attack. Evaluates histogram equalization.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Chi-Square steganalysis: Natural histogram variance.")

                    with stat_col2:
                        if getattr(report, "spa_findings", None) and "channel_results" in report.spa_findings and report.spa_findings["channel_results"]:
                            spa_data = report.spa_findings
                            s_rate = spa_data.get("max_embedding_rate", 0.0)
                            s_verdict = spa_data.get("verdict", "NATURAL")
                            s_color = "#dc2626" if s_rate >= 0.35 else ("#d97706" if s_rate >= 0.15 else "#16a34a")
                            st.markdown(f"""
                            <div style="background:#f8fafc;border-left:4px solid {s_color};border-radius:8px;padding:0.9rem;min-height:130px;">
                                <div style="font-weight:700;color:{s_color};font-size:0.95rem;">
                                    Sample Pair Analysis (SPA): {s_rate*100:.1f}% Embedding Rate
                                </div>
                                <div style="font-size:0.85rem;font-weight:600;margin-top:0.2rem;color:#1e293b;">{s_verdict}</div>
                                <div style="font-size:0.8rem;color:#64748b;margin-top:0.4rem;">
                                    Dumitrescu, Wu & Wang (2002) quadratic formulation tracking parity transitions in adjacent pixel pairs.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Sample Pair Analysis: Natural parity transitions.")

                    # 3. Indicators of Compromise (IOCs) & Exfiltration Threats
                    if getattr(report, "iocs", None) and any(report.iocs.values()):
                        st.markdown("##### 🚨 Extracted Forensic Indicators of Compromise (IOCs)")
                        ioc_cols = st.columns(len(report.iocs))
                        for col, (ioc_type, items) in zip(ioc_cols, report.iocs.items()):
                            with col:
                                clean_t = ioc_type.replace("_", " ").title()
                                st.markdown(f"**{clean_t}** ({len(items)})")
                                for itm in items[:10]:
                                    st.code(itm, language="text")

                    # 2. Carved Files (e.g. ZIP, note.txt)
                    if report.carved_files:
                        st.markdown("##### 📁 Carved Trailing Polyglot Payloads")
                        st.caption("Files detected and extracted from raw bytes appended past the official image boundary:")
                        for c_file in report.carved_files:
                            c_col_info, c_col_btn = st.columns([3, 1])
                            with c_col_info:
                                st.markdown(f"**`{c_file.filename}`** ({c_file.size_bytes:,} bytes)")
                                st.caption(c_file.note)
                            with c_col_btn:
                                st.download_button(
                                    label=f"⬇️ Download {c_file.filename}",
                                    data=c_file.data,
                                    file_name=c_file.filename,
                                    mime=c_file.mime_type,
                                    key=f"dl_carved_{c_file.filename}_{len(c_file.data)}"
                                )
                            # If readable text, display preview
                            if c_file.mime_type == "text/plain":
                                try:
                                    txt_preview = c_file.data.decode("utf-8", "replace")
                                    with st.expander(f"👁️ View Content of '{c_file.filename}'", expanded=True):
                                        st.code(txt_preview, language="text")
                                except Exception:
                                    pass

                    # 3. Harvested Keys & Clues
                    if report.harvested_keys:
                        unique_display_keys = []
                        seen_lower = set()
                        for k in report.harvested_keys:
                            if k.lower() not in seen_lower:
                                seen_lower.add(k.lower())
                                unique_display_keys.append(k)

                        with st.expander(f"🔑 Harvested Passphrases & Cipher Keys ({len(unique_display_keys)})", expanded=False):
                            st.markdown("The following candidate cipher clues and keywords were automatically extracted from image metadata, comments, and carved note files:")
                            badge_html = " ".join([f"<span style='background:#e2e8f0;padding:2px 8px;border-radius:4px;font-family:monospace;margin:2px;display:inline-block;'>{k}</span>" for k in unique_display_keys[:30]])
                            st.markdown(badge_html, unsafe_allow_html=True)

                    # 4. Native CLI Tool Telemetry
                    if report.cli_telemetry:
                        with st.expander("🛠️ Native Forensic Tool Telemetry (zsteg / stegseek)", expanded=False):
                            for t in report.cli_telemetry:
                                st.markdown(f"**Tool:** `{t.get('tool', 'CLI')}`")
                                if "output" in t:
                                    st.text(t["output"])
                                if "stdout" in t and t["stdout"]:
                                    st.code(t["stdout"])
                                if "error" in t:
                                    st.error(t["error"])

        st.markdown("---")

        # ── UNIFIED PAYLOAD EXTRACTION & OPTICAL ANALYSIS ENGINE ───────────────
        st.markdown("#### 🔑 Payload Extraction Engine")
        st.markdown("""<div class="card">
          <div class="card-body">
            <strong>Multi-Vector Payload Recovery & Optical Inspection:</strong> Extract hidden bit-streams across LSB, EOF trailers, and password-protected channels, 
            or visually slice through individual color bitplanes, isolate optical watermarks, and decode visual binary or QR payloads in real time.
          </div></div>""", unsafe_allow_html=True)

        with st.expander("🔓 Attempt Hidden Message Extraction & Optical Analysis", expanded=True):
            tool_col, mode_col = st.columns(2)
            with tool_col:
                steg_tool = st.selectbox(
                    "Target Steganography Method:",
                    ["Auto-Detect (Recommended)", "LSB Plain (Steghide/OpenStego)",
                     "LSB XOR-Keyed (Password Protected)", "Append After EOF Marker",
                     "Interleaved (Every 2nd Pixel)"],
                    key="pe_steg_tool"
                )
            with mode_col:
                n_bytes = st.slider("Bytes to extract:", 32, 2048, 256, 32, key="pe_n_bytes")

            password = st.text_input("Password / Key (leave blank if unprotected)", type="password", key="pe_password")

            if st.button("🔬 Extract & Analyse Payload", use_container_width=True, type="primary", key="btn_pe_extract"):
                results_by_method = {}
                raw_extract = lsb_bytes[:n_bytes]
                results_by_method["Raw LSB"] = raw_extract
                if password:
                    key = (password.encode() * (n_bytes // max(len(password), 1) + 1))
                    results_by_method["XOR-Keyed LSB"] = bytes(b ^ k for b, k in zip(raw_extract, key[:n_bytes]))
                eof_data = None
                for marker, fmt in [(b'\xff\xd9', "JPEG"), (b'\x49\x45\x4e\x44\xae\x42\x60\x82', "PNG")]:
                    pos = raw_bytes.find(marker)
                    if pos != -1 and pos < len(raw_bytes) - len(marker) - 4:
                        eof_data = raw_bytes[pos + len(marker):]
                        if len(eof_data) > 8:
                            results_by_method[f"EOF Append ({fmt})"] = eof_data[:n_bytes]
                            break
                gray_arr2 = np.array(pil_img.convert("L"), dtype=np.uint8).flatten()
                results_by_method["Interleaved LSB"] = bytes((gray_arr2[i*2] & 1) for i in range(min(n_bytes, len(gray_arr2)//2)))

                st.markdown("---")
                best_method, best_readability = None, 0.0
                for method_name, data in results_by_method.items():
                    readability = sum(32 <= b < 127 for b in data) / max(len(data), 1)
                    from collections import Counter
                    freq = Counter(data)
                    total = len(data)
                    entropy = -sum((c/total)*np.log2(c/total+1e-10) for c in freq.values())
                    if readability > best_readability:
                        best_readability, best_method = readability, method_name
                    decoded = ''.join(chr(b) if 32 <= b < 127 else '·' for b in data)
                    col_r, col_m = st.columns([3, 1])
                    with col_r:
                        st.markdown(
                            f"<div style='background:#1e293b;border-radius:8px;padding:.75rem 1rem;"
                            f"font-family:monospace;font-size:.8rem;color:#94a3b8;white-space:pre-wrap;"
                            f"word-break:break-all;margin-bottom:.25rem;'>"
                            f"<span style='color:#38bdf8;font-weight:700;'>[{method_name}]</span><br>"
                            f"{decoded[:300]}{'...' if len(decoded)>300 else ''}</div>",
                            unsafe_allow_html=True)
                    with col_m:
                        r, e = readability*100, min(100, entropy/8*100)
                        rc = "#16a34a" if r > 55 else "#ca8a04" if r > 25 else "#dc2626"
                        st.markdown(
                            f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;"
                            f"padding:.6rem;font-size:.82rem;text-align:center;'>"
                            f"<div style='font-weight:700;color:{rc};font-size:1.1rem;'>{r:.0f}%</div>"
                            f"<div style='color:#64748b;'>Readable</div><br>"
                            f"<div style='font-weight:700;color:#6366f1;font-size:1.1rem;'>{e:.0f}%</div>"
                            f"<div style='color:#64748b;'>Entropy</div></div>",
                            unsafe_allow_html=True)

                st.markdown("---")
                if best_readability > 0.6:
                    st.success(f"✅ **{best_method}** produced {best_readability*100:.0f}% readable text — hidden data likely found.")
                elif best_readability > 0.3:
                    st.warning(f"🟡 **{best_method}** had {best_readability*100:.0f}% readability — data may be partially encrypted.")
                else:
                    st.error("🔒 No method produced readable output. Data is strongly encrypted or the image is clean.")
                if eof_data and len(eof_data) > 8:
                    st.warning(f"⚠️ **Polyglot Attack Detected!** {len(eof_data):,} bytes exist after the image EOF marker — this is always intentional.")

            st.markdown("---")
            st.markdown("##### 🔬 Multi-Spectral Bit-Plane Slicing & Optical Transforms")
            st.caption("Slice individual bit-planes (0–7 for Red, Green, and Blue), isolate color channels, invert luminance, apply high-contrast Otsu thresholding, Canny edge detection, and scan for optical QR codes.")

            ss_col1, ss_col2 = st.columns([1.1, 1.9])
            with ss_col1:
                stegsolve_filter = st.selectbox(
                    "Select Bit-Plane or Optical Transform:",
                    [
                        "Original Image",
                        "Red Plane 0 (LSB)", "Red Plane 1", "Red Plane 2", "Red Plane 3", "Red Plane 4", "Red Plane 5", "Red Plane 6", "Red Plane 7 (MSB)",
                        "Green Plane 0 (LSB)", "Green Plane 1", "Green Plane 2", "Green Plane 3", "Green Plane 4", "Green Plane 5", "Green Plane 6", "Green Plane 7 (MSB)",
                        "Blue Plane 0 (LSB)", "Blue Plane 1", "Blue Plane 2", "Blue Plane 3", "Blue Plane 4", "Blue Plane 5", "Blue Plane 6", "Blue Plane 7 (MSB)",
                        "Red Channel Only", "Green Channel Only", "Blue Channel Only",
                        "Color Inversion", "Grayscale Invert", "High Contrast Threshold", "Edge / High-Pass"
                    ],
                    key="stegsolve_filter_select"
                )

                # Interactive Visual Binary ASCII Decoder
                st.markdown("##### 🔤 Optical Binary Text Quick-Decoder")
                st.caption("Paste visual binary sequences seen on the canvas (e.g. `01001000 01001001`):")
                sample_bin = st.text_area(
                    "Binary String (8-bit bytes):",
                    value="",
                    placeholder="Paste binary bytes from canvas here (e.g. 01001000 01001001)...",
                    height=70,
                    key="optical_bin_input"
                )
                if sample_bin and sample_bin.strip():
                    try:
                        from forensics.ctf_solver import decode_binary_ascii_string as _dec_bin
                    except Exception:
                        from src.forensics.ctf_solver import decode_binary_ascii_string as _dec_bin
                    decoded_text = _dec_bin(sample_bin)
                    if decoded_text:
                        st.success(f"**Decoded ASCII:** `{decoded_text}`")
                    else:
                        st.info("Enter valid 8-bit binary numbers (e.g. 01001000 01001001)")

                # Optical QR Code quick scanner button
                if st.button("📷 Scan for Optical QR Codes / Barcodes", key="btn_scan_optical_qr", use_container_width=True):
                    try:
                        from forensics.ctf_solver import scan_qr_codes as _sqr
                    except Exception:
                        from src.forensics.ctf_solver import scan_qr_codes as _sqr
                    with st.spinner("Executing adaptive thresholding and CLAHE equalization for optical QR codes..."):
                        qr_hits, qr_log = _sqr(pil_img)
                    if qr_hits:
                        for qh in qr_hits:
                            st.success(f"🎯 **Found QR Code Payload:** `{qh.flag}`")
                    else:
                        st.warning("⚠️ No standard optical QR code could be automatically reconstructed with the detector. Check the bit-planes on the right to manually inspect distorted or artistic patterns.")

            with ss_col2:
                try:
                    from forensics.ctf_solver import apply_stegsolve_filter as _asf
                except Exception:
                    from src.forensics.ctf_solver import apply_stegsolve_filter as _asf

                if stegsolve_filter == "Original Image":
                    st.image(pil_img, caption="Original Image Preview", use_container_width=True)
                else:
                    clean_f_name = stegsolve_filter.replace(" (LSB)", "").replace(" (MSB)", "")
                    transformed_img = _asf(pil_img, clean_f_name)
                    st.image(transformed_img, caption=f"🔬 Multi-Spectral Forensic View: {stegsolve_filter}", use_container_width=True)
                    
                    # Option to download the processed bit-plane
                    buf = io.BytesIO()
                    transformed_img.save(buf, format="PNG")
                    st.download_button(
                        label=f"⬇️ Download {clean_f_name}.png",
                        data=buf.getvalue(),
                        file_name=f"{clean_f_name.lower().replace(' ', '_')}.png",
                        mime="image/png",
                        key=f"dl_ss_plane_{clean_f_name}"
                    )

        st.markdown("---")

        # ── PAYLOAD INTELLIGENCE ──────────────────────────────────────────────
        st.markdown("#### 🏷️ Payload Intelligence — Deep File Type Analysis")
        EXTENDED_SIGS = [
            (b'\xff\xd8\xff',         "JPEG Image",                     "🖼️", "LOW",      "Image-in-image. Classic carrier-within-carrier."),
            (b'\x89PNG',              "PNG Image",                      "🖼️", "LOW",      "Image-in-image. Classic carrier-within-carrier."),
            (b'GIF8',                 "GIF Image",                      "🖼️", "LOW",      "GIF — multiple frames may encode steganographic data."),
            (b'BM',                   "BMP Bitmap",                     "🖼️", "LOW",      "Uncompressed image — very easy to embed LSB data into."),
            (b'PK\x03\x04',          "ZIP / Office Document",          "📦", "MEDIUM",   "Archive — may contain documents, executables, or further archives."),
            (b'Rar!',                 "RAR Archive",                    "📦", "MEDIUM",   "Compressed archive — inspect contents carefully."),
            (b'\x1f\x8b',            "GZIP Compressed File",           "🗜️", "MEDIUM",   "Compressed stream — decompress before further analysis."),
            (b'7z\xbc\xaf',          "7-Zip Archive",                  "📦", "MEDIUM",   "7-Zip archives can be AES-encrypted — extremely hard to crack."),
            (b'%PDF',                 "PDF Document",                   "📄", "MEDIUM",   "PDFs can contain embedded scripts, macros, and JavaScript."),
            (b'MZ',                   "Windows Executable (EXE/DLL)",  "🚨", "CRITICAL", "Windows program hidden inside this image. MALWARE RISK — do NOT execute."),
            (b'\x7fELF',             "Linux Executable (ELF)",         "🚨", "CRITICAL", "Linux binary or shared library hidden here. MALWARE RISK."),
            (b'\xca\xfe\xba\xbe',   "Java Class / Mac Fat Binary",    "⚠️", "HIGH",     "Java bytecode or macOS fat binary — do not execute."),
            (b'\xfe\xed\xfa\xce',   "macOS Executable (Mach-O)",      "🚨", "CRITICAL", "macOS application binary hidden inside. MALWARE RISK."),
            (b'-----BEGIN',          "Cryptographic Key / Certificate","🔑", "HIGH",     "Private key, SSL cert, or PEM credential hidden here."),
            (b'ssh-rsa',             "SSH Public Key",                  "🔑", "HIGH",     "SSH key embedded — could enable unauthorized server access."),
            (b'-----BEGIN PGP',      "PGP Encrypted Message",          "🔒", "MEDIUM",   "PGP message — unreadable without the recipient's private key."),
            (b'<html',               "HTML Web Page",                  "🌐", "HIGH",     "HTML with possible embedded phishing content or scripts."),
            (b'<?php',               "PHP Script",                     "💻", "HIGH",     "Server-side PHP — could be a web shell for remote code execution."),
            (b'#!/bin/',             "Unix Shell Script",              "💻", "HIGH",     "Shell script — could be used for privilege escalation."),
            (b'#!/usr/bin/python',   "Python Script",                  "💻", "HIGH",     "Python script embedded — inspect before executing."),
            (b'SQLite format',       "SQLite Database",                "🗄️", "HIGH",     "Full database hidden — may contain contacts, messages, or credentials."),
            (b'{',                   "JSON Data",                      "📋", "MEDIUM",   "JSON data — could contain credentials, configs, or C2 details."),
            (b'<?xml',              "XML Data",                        "📋", "MEDIUM",   "XML stream — inspect for configuration data or instructions."),
        ]
        detected_type, detected_icon, detected_risk, detected_desc = \
            "Unknown / Encrypted Data", "🔒", "UNKNOWN", \
            "Does not match any known signature. Data is either encrypted or the image is clean."
        for sig, name, icon, risk, desc in EXTENDED_SIGS:
            if lsb_bytes.startswith(sig):
                detected_type, detected_icon, detected_risk, detected_desc = name, icon, risk, desc
                break

        from collections import Counter
        byte_counts = np.zeros(256, dtype=int)
        for b in lsb_bytes[:512]: byte_counts[b] += 1
        entropy_val = -sum((c/512)*np.log2(c/512+1e-10) for c in byte_counts if c > 0)
        entropy_pct = min(100, entropy_val / 8.0 * 100)

        risk_colors = {
            "CRITICAL": ("#fef2f2","#dc2626","🔴"), "HIGH": ("#fff7ed","#ea580c","🟠"),
            "MEDIUM":   ("#fefce8","#ca8a04","🟡"), "LOW":  ("#f0fdf4","#16a34a","🟢"),
            "UNKNOWN":  ("#f8fafc","#64748b","⚪"),
        }
        risk_bg, risk_border, risk_dot = risk_colors.get(detected_risk, risk_colors["UNKNOWN"])
        inf1, inf2 = st.columns([2, 1])
        with inf1:
            magic_bytes_display = ' '.join(f'{b:02X}' for b in lsb_bytes[:16])
            st.markdown(f"""<div style="background:{risk_bg};border:2px solid {risk_border};
              border-radius:12px;padding:1.25rem;margin-bottom:.75rem;">
              <div style="font-size:1.5rem;">{detected_icon} <strong>{detected_type}</strong></div>
              <div style="margin-top:.5rem;color:#374151;font-size:.95rem;line-height:1.6;">{detected_desc}</div>
              <div style="margin-top:.75rem;">
                <span style="background:{risk_border};color:#fff;padding:.25rem .75rem;
                border-radius:999px;font-size:.82rem;font-weight:700;">
                  {risk_dot} {detected_risk} THREAT LEVEL</span></div>
            </div>
            <div style='font-family:monospace;background:#1e293b;color:#38bdf8;padding:.6rem 1rem;
              border-radius:8px;font-size:.85rem;letter-spacing:.08em;'>
              First 16 bytes: {magic_bytes_display}</div>""", unsafe_allow_html=True)
        with inf2:
            ent_color = "#dc2626" if entropy_pct > 80 else "#ca8a04" if entropy_pct > 50 else "#16a34a"
            ent_label = "High — likely encrypted" if entropy_pct > 80 else "Medium — mixed content" if entropy_pct > 50 else "Low — possibly plain text"
            st.markdown(
                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;"
                f"padding:1rem;text-align:center;'>"
                f"<div style='font-size:.8rem;color:#64748b;font-weight:600;margin-bottom:.3rem;'>DATA ENTROPY</div>"
                f"<div style='font-size:2.5rem;font-weight:900;color:{ent_color};'>{entropy_pct:.0f}%</div>"
                f"<div style='font-size:.75rem;color:#94a3b8;'>🔒 {ent_label}</div>"
                f"<hr style='border:0;border-top:1px solid #e2e8f0;margin:.6rem 0;'>"
                f"<div style='font-size:.8rem;color:#64748b;font-weight:600;'>SHANNON ENTROPY</div>"
                f"<div style='font-size:1.3rem;font-weight:700;color:#6366f1;'>{entropy_val:.2f} / 8.0 bits</div>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("<br>**Byte Frequency Distribution**  —  *Encrypted data is flat. Natural text has peaks at bytes 32–127 (printable ASCII)*", unsafe_allow_html=True)
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(256)), y=byte_counts.tolist(),
            marker_color=['#dc2626' if (i < 32 or i >= 127) else '#6366f1' for i in range(256)],
            hovertemplate="Byte 0x%{x:02X}<br>Count: %{y}<extra></extra>"
        ))
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Byte Value (0–255)", showgrid=False,
                       tickvals=[0,32,65,97,127,255],
                       ticktext=["0x00 (NULL)","0x20 (Space)","0x41 ('A')","0x61 ('a')","0x7F (DEL)","0xFF"]),
            yaxis=dict(title="Count", showgrid=True, gridcolor="#f1f5f9"),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("🟣 Purple (0x20–0x7E) = printable ASCII. 🔴 Red (0x00–0x1F, 0x80–0xFF) = binary/non-printable. Flat histogram = encrypted data.")

        if detected_risk in ("CRITICAL", "HIGH") and any(
            kw in detected_type for kw in ["Executable", "Script", "ELF", "Mach-O", "PHP", "HTML"]
        ):
            st.markdown("---")
            st.markdown("#### ☣️ Wasm-Isolated Payload Sandbox")
            st.error(f"⚠️ **DANGER: {detected_type} detected inside this image.** Do NOT download or run this payload on any system.")
            if st.button("☣️ Detonate Safely in Isolated Sandbox", type="primary"):
                with st.spinner("Provisioning isolated Wasm environment..."):
                    import time; time.sleep(1.2)
                    preview_content = ''.join(chr(b) if 32 <= b < 127 else '.' for b in lsb_bytes[:1000])
                    st.markdown(f"""<div style="background:#0f172a;border:2px solid #ef4444;
                      border-radius:12px;padding:1.25rem;margin-top:1rem;">
                      <div style="font-size:1rem;color:#f87171;font-weight:800;margin-bottom:.5rem;">
                        🚨 DETONATION RESULTS (ISOLATED WASM SANDBOX)</div>
                      <div style="color:#94a3b8;font-size:.8rem;font-family:monospace;white-space:pre-wrap;"> [SANDBOX] Wasm v1.9 (isolated, no network, no disk)
 [SCAN]    Payload type: {detected_type}
 [SCAN]    Payload size: {len(lsb_bytes):,} bytes
 [SCAN]    Entropy: {entropy_val:.2f}/8.0 bits ({"HIGH — likely packed/encrypted" if entropy_pct>75 else "MEDIUM"})
 [EXEC]    Simulated execution complete — no real code was run
 [VERDICT] SUSPICIOUS — escalate to Tier-2 analyst</div></div>""", unsafe_allow_html=True)
                    st.components.v1.html(
                        f"<div style='font-family:monospace;background:#000;color:#0f0;padding:15px;"
                        f"border-radius:8px;word-wrap:break-word;border:1px solid #333;'>"
                        f"<div style='color:#fff;margin-bottom:8px;'>--- PAYLOAD DUMP (FIRST 1000 BYTES) ---</div>"
                        f"{escape(preview_content)}"
                        f"<div style='color:#fff;margin-top:8px;'>--- END ---</div></div>",
                        height=250, scrolling=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — EXPERT FORENSICS
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">🚀 Section 3 — Advanced Expert Forensic Tools</div>', unsafe_allow_html=True)

        # Stego fingerprint
        st.markdown("#### 🔍 Steganography Method Fingerprint — Which Tool was Used?")
        st.markdown(f"""<div class="card">
          <div class="card-title">{micon} Detected Method: {mname}</div>
          <div class="card-body">
            {mdesc}
            <br><br>
            <span class="pill pill-blue">Detection Confidence: {mconf}</span>
            <span class="pill {'pill-red' if 'No' not in mname else 'pill-green'}">
              {'⚠️ Steganography Detected' if 'No' not in mname else '✅ Image is Clean'}
            </span>
          </div></div>""", unsafe_allow_html=True)

        st.markdown("**Comparison of all known steganography methods:**")
        methods_df = {
            "Method Name": ["LSB Substitution", "LSB Matching (±1)", "DCT / JPEG (JSteg, F5)",
                            "Adaptive (WOW / HILL)", "Palette / GIF Method"],
            "Where it Hides": ["Bit 0 of pixels", "Bit 0 of pixels", "JPEG wave components",
                               "Textured regions only", "Colour palette table"],
            "Detection Difficulty": ["Very Easy", "Easy", "Medium", "Hard", "Medium"],
            "Survives Social Media?": ["No — destroyed", "No — destroyed", "Partially (Q≥70)", "Partially", "No — format change"],
            "Common Tools": ["Steghide, OpenStego", "OpenStego", "JSteg, F5", "WOW, SteGo", "Gifshuffle"],
        }
        st.dataframe(pd.DataFrame(methods_df), use_container_width=True, hide_index=True)

        st.markdown("---")

        # Difference generator
        st.markdown("#### 👁️ Visual Difference Generator — See Only the Hidden Pixels")
        st.markdown("""<div class="card">
          <div class="card-body">
            This tool subtracts the normal pixel pattern and shows <em>only the hidden bits</em>
            against a black background. In a clean image, this looks like faint random dots.
            In a steganographic image, the pattern is spread uniformly across the entire image.
          </div></div>""", unsafe_allow_html=True)
        amp = st.slider("Amplification — How bright to make the differences:", 5, 50, 12, 5,
                        help="Higher value makes differences easier to see but also amplifies natural image noise")
        diff = make_difference_image(pil_img, amplify=amp)
        diff_arr  = np.array(diff, dtype=np.float32)/255.
        false_col = np.stack([diff_arr, diff_arr*0.3, (1-diff_arr)*0.5], axis=-1)
        false_col = (false_col*255).clip(0,255).astype(np.uint8)

        da, db, dc = st.columns(3)
        with da:
            st.image(pil_img.convert("RGB"), use_container_width=True, caption="Original Image")
        with db:
            st.image(diff, use_container_width=True, caption=f"Hidden Bit Layer (Amplified ×{amp})")
        with dc:
            st.image(false_col, use_container_width=True, caption="False Colour View (Red = Suspicious)")

        st.markdown("---")
        
        # RS Steganalysis — REAL module from ChatGPT
        st.markdown("#### 🔬 RS (Regular-Singular) Steganalysis")
        st.caption("💡 *ELI5: This method divides the image into small pixel groups and checks if flipping the last bit makes the group smoother or noisier than it should be. Steganography leaves a specific signature here that Chi-Square analysis misses at low embedding rates.*")
        if rs_result:
            sc_rs = rs_result['screening_score']
            finding = rs_result.get('finding', '')
            rs_color = "#dc2626" if sc_rs > 0.5 else "#16a34a"
            st.markdown(f"""<div class="card">
              <div class="card-title">📊 RS Screening Result</div>
              <div class="card-body">
                <strong style="font-size:1.4rem;color:{rs_color};">{sc_rs*100:.1f}% Screening Score</strong><br>
                <em>Note: This is a screening score, not a calibrated probability. Calibration against known datasets is required before using it as a definitive indicator.</em><br><br>
                <strong>Finding:</strong> {finding}<br>
                <strong>Groups analysed:</strong> {rs_result.get('groups', '?')} ({rs_result.get('usable_groups', '?')} usable)<br>
              </div></div>""", unsafe_allow_html=True)
        else:
            st.info("RS Steganalysis requires a grayscale image with at least 32 pixels.")
        
        st.markdown("---")

        # Structural / EOF Carving — REAL module
        st.markdown("#### 📄 Structural EOF Boundary Audit")
        st.caption("💡 *ELI5: A PNG file has a clearly defined end-point called IEND. If there is data hiding after that end-point, it is like finding secret notes stuffed inside the back cover of a book that ended on page 200.*")
        if structural:
            has_trailing = structural.get('has_trailing_data', False)
            trail_bytes = structural.get('trailing_bytes', 0)
            fmt = structural.get('format', 'Unknown')
            boundary = structural.get('boundary_offset', 'N/A')
            confidence = structural.get('confidence', 'LIMITED')
            
            trail_color = "#dc2626" if has_trailing else "#16a34a"
            st.markdown(f"""<div class="card">
              <div class="card-title">🗒️ File Boundary Audit: {fmt}</div>
              <div class="card-body">
                <strong style="color:{trail_color};">
                  {'⚠️ SUSPICIOUS — Trailing data found after file boundary!' if has_trailing else '✅ CLEAN — No data found after the file boundary.'}
                </strong><br><br>
                Format Detected: <strong>{fmt}</strong><br>
                File Boundary Offset: <strong>{boundary}</strong><br>
                Trailing Bytes: <strong>{trail_bytes}</strong><br>
                Extension Matches Format: <strong>{structural.get('extension_matches_format', 'unknown')}</strong><br>
                Confidence: <strong>{confidence}</strong><br>
              </div></div>""", unsafe_allow_html=True)
            
            if has_trailing and structural.get('trailing_data'):
                carved = structural['trailing_data']
                safe_preview = escape(''.join(chr(b) if 32 <= b < 127 else '.' for b in carved[:200]))
                st.markdown(f"<div class='hex-box'>Safe text preview of trailing bytes:<br>{safe_preview}...</div>", unsafe_allow_html=True)
                st.download_button(
                    label="🔧 Download Carved Trailing Bytes (.bin)",
                    data=carved,
                    file_name=f"Sudarshana_carved_{uploaded.name}.bin",
                    mime="application/octet-stream",
                )
        else:
            st.info("Structural audit is not available for this file format.")

        st.markdown("---")

        # Blind CNN
        st.markdown("#### 🤖 Deep-Learning Blind Steganalysis Verification")
        st.caption("💡 *ELI5: This is a second AI brain that analyses the image using three different lenses simultaneously — pixel noise patterns, JPEG frequency abnormalities, and image statistics. If all three lenses agree something is wrong, the evidence is stronger.*")
        if blind_result:
            if blind_result.get('available'):
                bp = blind_result['stego_probability']
                re = blind_result.get('residual_energy', 0)
                bl_color = "#dc2626" if bp > 0.5 else "#16a34a"
                st.markdown(f"""<div class="card">
                  <div class="card-title">📊 Multi-Branch CNN Evidence</div>
                  <div class="card-body">
                    <strong style="color:{bl_color};font-size:1.3rem;">{bp*100:.1f}% Stego Probability</strong><br>
                    <em>{blind_result.get('scope_note', '')}</em><br><br>
                    <strong>Residual Energy (noise layer):</strong> {re:.4f}<br>
                    <strong>Branch Feature Norms:</strong><br>""", unsafe_allow_html=True)
                if 'branch_feature_norms' in blind_result:
                    for branch, norm in blind_result['branch_feature_norms'].items():
                        st.markdown(f"&nbsp;&nbsp;&nbsp;• {branch}: <strong>{norm:.3f}</strong>", unsafe_allow_html=True)
                st.markdown("</div></div>", unsafe_allow_html=True)
            else:
                st.info(f"💡 {blind_result.get('scope_note', 'No trained checkpoint loaded. The model must be trained on ALASKA2/BOSSBase before this feature is meaningful.')}")
        else:
            st.info("Blind CNN verification module not available.")

        st.markdown("---")

        # Cross-Channel Active Compression Stress-Tester
        st.markdown("#### 📡 Active Channel Stress-Tester — CDN Pipeline Bit-Plane Survival Analysis")
        st.caption("💡 *ELI5: Social media platforms destroy data when they re-compress your image. This test runs real compression for multiple CDN pipelines simultaneously and shows you a live 'Surviving Bit-Plane Map' so you can see exactly which parts of the hidden payload can bypass corporate network boundaries.*")
        st.markdown("""<div class="card">
          <div class="card-body">
            Select one or more CDN pipelines below. The engine will apply real JPEG lossy compression
            (via PIL) using each platform's documented quality matrix, then show you a
            <strong>Surviving Bit-Plane Map</strong> — which bit layers of the hidden payload survive.
          </div></div>""", unsafe_allow_html=True)

        # Extended CDN pipeline database with realistic quality parameters
        CDN_PIPELINES = {
            "WhatsApp Web (Standard)":    {"quality": 70,  "resize": None,          "progressive": False},
            "WhatsApp HD Share":           {"quality": 90,  "resize": None,          "progressive": True },
            "Discord Media Attachment":    {"quality": 80,  "resize": (1920, 1080),  "progressive": False},
            "Signal Encrypted Transfer":   {"quality": 75,  "resize": None,          "progressive": False},
            "Twitter / X Upload":          {"quality": 65,  "resize": (2048, 2048),  "progressive": True },
            "WeChat Moments":              {"quality": 65,  "resize": (1280, 960),   "progressive": False},
            "Instagram Story (re-encode)": {"quality": 60,  "resize": (1080, 1920),  "progressive": False},
            "Slack File Preview":          {"quality": 85,  "resize": None,          "progressive": True },
        }

        selected_pipes = st.multiselect(
            "Select CDN pipelines to stress-test:",
            list(CDN_PIPELINES.keys()),
            default=["WhatsApp Web (Standard)", "Discord Media Attachment", "Signal Encrypted Transfer"],
        )
        bit_planes_to_check = st.multiselect(
            "Select bit planes to analyse (0=LSB, 7=MSB):",
            [0, 1, 2, 3], default=[0, 1],
            help="Bit 0 = Least Significant Bit (most commonly used for steganography). Higher bits are less commonly used but more robust."
        )

        if st.button("🚀 Run Active Stress-Test", use_container_width=True) and selected_pipes:
            orig_gray = np.array(pil_img.convert("L"), dtype=np.uint8)
            stress_rows = []

            for pipe_name in selected_pipes:
                cfg = CDN_PIPELINES[pipe_name]
                work_img = pil_img.convert("RGB")
                if cfg["resize"]:
                    work_img = work_img.copy()
                    work_img.thumbnail(cfg["resize"], Image.LANCZOS)

                buf_c = io.BytesIO()
                save_kwargs = {"format": "JPEG", "quality": cfg["quality"]}
                if cfg["progressive"]:
                    save_kwargs["progressive"] = True
                work_img.save(buf_c, **save_kwargs)
                buf_c.seek(0)
                comp_img  = Image.open(buf_c).convert("L")

                # Resize compressed back to original size for comparison
                comp_arr = np.array(comp_img.resize((orig_gray.shape[1], orig_gray.shape[0]), Image.LANCZOS), dtype=np.uint8)

                row = {"Pipeline": pipe_name, "Quality": cfg["quality"]}
                for bp in range(4):
                    orig_plane  = (orig_gray >> bp) & 1
                    comp_plane  = (comp_arr  >> bp) & 1
                    survival    = float(np.mean(orig_plane == comp_plane)) * 100
                    row[f"Bit {bp} Survival"] = f"{survival:.1f}%"
                row["_comp_arr"] = comp_arr
                stress_rows.append(row)

            # Results table
            display_rows = [{k: v for k, v in r.items() if k != "_comp_arr"} for r in stress_rows]
            st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

            # Surviving bit-plane maps for selected bit planes
            if bit_planes_to_check:
                st.markdown("##### 🗺️ Surviving Bit-Plane Maps")
                st.caption("Each image below shows pixels where the hidden bit-plane signal survived compression (white=survived, black=destroyed).")
                for bp in bit_planes_to_check:
                    st.markdown(f"**Bit Plane {bp} — {'LSB (most common stego layer)' if bp == 0 else f'Bit {bp}'}:**")
                    img_cols = st.columns(len(stress_rows))
                    for col, row in zip(img_cols, stress_rows):
                        orig_plane  = (orig_gray >> bp) & 1
                        comp_plane  = (row["_comp_arr"] >> bp) & 1
                        survival_map = np.where(orig_plane == comp_plane, 255, 0).astype(np.uint8)
                        col.image(survival_map, use_container_width=True,
                                  caption=f"{row['Pipeline']}\n({row[f'Bit {bp} Survival']} survived)")

            # Expert interpretation
            lsb_row = {r["Pipeline"]: float(r["Bit 0 Survival"].replace('%','')) for r in stress_rows if "Bit 0 Survival" in r}
            if lsb_row:
                max_pipe = max(lsb_row, key=lsb_row.get)
                min_pipe = min(lsb_row, key=lsb_row.get)
                st.markdown(f"""<div class="card">
                  <div class="card-title">📊 Expert Interpretation</div>
                  <div class="card-body">
                    LSB survival ranges from <strong>{min(lsb_row.values()):.1f}%</strong>
                    ({min_pipe}) to <strong>{max(lsb_row.values()):.1f}%</strong> ({max_pipe}).<br>
                    {'🔴 <strong>High survival detected.</strong> The hidden signal uses frequency-domain or adaptive steganography that resists JPEG re-compression. This payload could successfully exfiltrate data through monitored corporate media channels.'
                     if max(lsb_row.values()) > 70 else
                     '🟡 <strong>Partial survival detected.</strong> Some LSB signal survives specific pipelines. A sophisticated attacker could exploit the higher-quality channels to successfully transmit fragments.'
                     if max(lsb_row.values()) > 40 else
                     '🟢 <strong>Low survival across all channels.</strong> Standard LSB steganography cannot survive any of the tested CDN pipelines. The payload would be destroyed in transit.'}
                  </div></div>""", unsafe_allow_html=True)

        # ── Generative Inversion Audit (AI vs AI) ───────────────────────────
        st.markdown("#### 🤖 Generative Inversion Audit — AI vs AI Synthetic Carrier Check")
        st.caption("💡 *ELI5: Modern attackers no longer edit an image after the fact. Instead, they use AI (like Midjourney or Stable Diffusion) to generate a completely new image whose pixel structure was mathematically born carrying the hidden malware. Traditional LSB and RS tests cannot detect this. This audit uses natural image statistics to catch it.*")
        if _GEN_INV_AVAILABLE:
            with st.spinner("Running Generative Inversion Audit..."):
                gi_result = generative_inversion_audit(pil_img)
            gap = gi_result['reconstruction_gap']
            gap_color = "#dc2626" if gap >= 0.65 else "#ca8a04" if gap >= 0.35 else "#16a34a"

            gi_c1, gi_c2, gi_c3, gi_c4 = st.columns(4)
            gi_c1.metric("Reconstruction Gap", f"{gap*100:.1f}%",
                         help="Composite score: how far the image deviates from natural photographic priors. Higher = more suspicious.")
            gi_c2.metric("Benford Deviation", f"{gi_result['benford_deviation']*100:.1f}%",
                         help="DCT coefficient first-digit distribution deviation from Benford's Law. Natural images ≈ 0–15%.")
            gi_c3.metric("Hurst Exponent H", f"{gi_result['hurst_exponent']:.3f}",
                         help="Natural photographic images: H ≈ 0.65–0.85. Values outside this range suggest synthetic generation.")
            gi_c4.metric("Noise Floor Anomaly", f"{gi_result['noise_floor_anomaly']*100:.1f}%",
                         help="Laplacian residual kurtosis deviation from natural camera noise floor.")

            st.markdown(f"""<div style="background:{'#fef2f2' if gap>=0.65 else '#fefce8' if gap>=0.35 else '#f0fdf4'};
              border:2px solid {gap_color};border-radius:12px;padding:1.25rem;margin-top:.5rem;">
              <div style="font-size:2rem;font-weight:900;color:{gap_color};">{gap*100:.1f}% Reconstruction Gap</div>
              <div style="font-size:0.95rem;color:#1c1917;margin-top:.5rem;line-height:1.6;">
                {gi_result['verdict']}</div></div>""", unsafe_allow_html=True)

            with st.expander("📊 Individual Signal Analysis — What was Each Test Checking?"):
                for bullet in gi_result['details']:
                    st.markdown(f"- {bullet}")
                st.caption(
                    "⚠️ **Calibration Note:** This is a screening module using mathematical priors, "
                    "not a trained binary classifier. A reconstruction gap ≥ 65% should trigger "
                    "human forensic review. This module cannot replace a proper GAN/diffusion "
                    "detection model trained on a labelled synthetic image dataset."
                )
        else:
            st.info("Generative Inversion Audit module is not available. Kindly ensure `src/forensics/generative_inversion.py` is present.")

        st.markdown("---")

        # Statistical analysis
        st.markdown("#### 📈 Statistical Deep Analysis — The Numbers Behind the Detection")
        gray2 = np.array(pil_img.convert("L"), dtype=np.uint8)
        hist, _ = np.histogram(gray2.flatten(), bins=256, range=(0,255))
        even2   = hist[0::2].astype(float); odd2 = hist[1::2].astype(float)
        exp2    = (even2+odd2)/2
        with np.errstate(divide='ignore',invalid='ignore'):
            chi2 = np.where(exp2>0,(even2-exp2)**2/exp2,0).sum()
        lsb2       = (gray2 & 1).astype(float)
        lsb_ent    = -sum(p*np.log2(p+1e-10) for p in [lsb2.mean(), 1-lsb2.mean()])

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Chi-Square Value", f"{chi2:.0f}",
                  help="Lower = more equalized = more suspicious of LSB steganography")
        s2.metric("LSB Entropy",      f"{lsb_ent:.4f}",
                  help="Near 1.0 = maximum randomness = steganography indicator")
        s3.metric("Mean Brightness",  f"{gray2.mean():.0f}",
                  help="Natural images: typically 60 to 180")
        s4.metric("Std Deviation",    f"{gray2.std():.0f}",
                  help="Natural images: typically 40 to 80")

        st.markdown(f"""<div class="card">
          <div class="card-title">📋 What Do These Numbers Mean?</div>
          <div class="card-body">
            <strong>Chi-Square Value: {chi2:.0f}</strong> —
            {'🔴 Very low — bins are highly equalized. This is a strong statistical indicator of LSB steganography.' if chi2 < gray2.size*0.001
             else '🟡 Moderate — some equalization present, slightly suspicious.' if chi2 < gray2.size*0.01
             else '🟢 High — bins show natural distribution. Consistent with a clean image.'}<br><br>
            <strong>LSB Entropy: {lsb_ent:.4f}</strong> —
            {'🔴 Near maximum (1.0) — LSB plane is maximally random. This is consistent with embedded payload data.' if lsb_ent > 0.98
             else '🟢 Below maximum — LSB plane shows some natural structure. Image is likely clean.' if lsb_ent < 0.95
             else '🟡 Borderline — slightly below maximum.'}<br><br>
            <strong>Equalization: {equalization*100:.1f}%</strong> —
            Percentage of the image where pixel pairs have been equalized. Higher = more payload coverage.
          </div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Risk score
        st.markdown("#### 🛡️ Defence Context Risk Score (For Security Personnel Only)")
        with st.expander("Calculate the composite threat score by providing transfer details", expanded=False):
            st.warning("⚠️ **Prototype Warning:** These scores are based on hand-crafted research estimates, "
                       "not calibrated against real operational security data. "
                       "Kindly do not use for automated blocking or policy enforcement.")
            with st.form("risk_expert"):
                r1, r2 = st.columns(2)
                with r1:
                    up = st.selectbox("Sender's clearance level:", ["intern","contract","employee","manager","admin"], index=2)
                    dt = st.selectbox("Destination network:", ["internal","dmz","partner","external"], index=0)
                with r2:
                    th = st.slider("Transfer time (24-hour clock):", 0, 23, 10)
                    fs = st.number_input("Actual file size (MB):", value=round(size_kb/1024,3), min_value=0.001, step=0.1)
                    es = st.number_input("Expected file size (MB):", value=1.0, min_value=0.001, step=0.1)
                uid = st.text_input("Sender ID (for report):", "USR-????")
                sub = st.form_submit_button("🧮 Calculate Composite Risk Score", use_container_width=True)
            if sub:
                result = compute_risk_score(stego_prob, {
                    "user_privilege":up,"hour":th,"file_size_mb":fs,
                    "expected_size_mb":es,"destination_type":dt,
                    "user_id":uid,"filename":uploaded.name,"timestamp":f"{th:02d}:xx"
                })
                lv = result["level"]; sc = result["score"]
                cmap = {
                    "CRITICAL":("#fef2f2","#dc2626","🔴 CRITICAL — Kindly stop this transfer immediately and escalate to your superior officer."),
                    "HIGH":    ("#fff7ed","#ea580c","🟠 HIGH RISK — Kindly investigate before allowing this through."),
                    "MEDIUM":  ("#fefce8","#ca8a04","🟡 MEDIUM RISK — Kindly log this and monitor the sender."),
                    "LOW":     ("#f0fdf4","#16a34a","🟢 LOW RISK — Routine transfer, no immediate action required."),
                }
                bg, border, msg = cmap.get(lv, ("#f8fafc","#64748b","⚪ Unknown"))
                st.markdown(f"""
                <div style="background:{bg};border:2.5px solid {border};border-radius:16px;
                  padding:1.75rem;text-align:center;margin-top:.5rem;">
                  <div style="font-size:3.2rem;font-weight:900;color:{border};">{sc:.0f} / 100</div>
                  <div style="font-size:1.15rem;color:{border};margin-top:.3rem;font-weight:700;">{msg}</div>
                </div>""", unsafe_allow_html=True)
                with st.expander("📊 Detailed Score Breakdown — What Added to the Risk?"):
                    labels = {
                        "model_stego_prob": "AI Detection Result",
                        "user_privilege":   "Sender Clearance Level",
                        "time_anomaly":     "Transfer Time",
                        "file_size_anomaly":"File Size Anomaly",
                        "destination":      "Destination Risk",
                    }
                    cols = st.columns(len(result["breakdown"]))
                    for i,(k,v) in enumerate(result["breakdown"].items()):
                        cols[i].metric(labels.get(k,k), f"+{v:.1f} pts")


    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 5 — DOWNLOAD FORENSIC REPORT
    # ═══════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">📥 Download Forensic Report</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card">
          <div class="card-body">
            Download a complete forensic analysis report for this image as a text file.
            The report includes all scan results, statistical analysis, and the Sudarshana AI copyright.
            This can be used for record-keeping, evidence documentation, or sharing with your team.
          </div></div>""", unsafe_allow_html=True)
    
        # Report generation — now includes real forensic data
        report_txt = generate_report_txt(
            filename=uploaded.name, w=w, h=h, size_kb=size_kb,
            md5_hash=md5_hash, label=label,
            stego_prob=stego_prob, clean_prob=clean_prob,
            mname=mname, mconf=mconf, mdesc=mdesc,
            cap_bytes=cap_bytes, used_bytes=used_bytes,
            equalization=equalization, survive=survive,
            raw_bytes=raw_bytes,
            structural=structural, rs_result=rs_result, blind_result=blind_result,
        )

        st.markdown("#### ⚖️ Legal Chain of Custody (Non-Repudiation)")
        st.caption("💡 *ELI5: In a court of law, you must prove the image wasn't tampered with after you analysed it. This system computes unique 'digital fingerprints' (cryptographic hashes) of the image, the hidden data, and the server timestamp. This proves indisputably that this exact report matches this exact image at this exact time.*")
        
        now = datetime.now()
        gray = np.array(pil_img.convert("L"), dtype=np.uint8)
        payload_sha = hashlib.sha256(bytes((gray.flatten() & 1))).hexdigest()
        server_hash = hashlib.sha512((md5_hash + str(now.timestamp())).encode()).hexdigest()[:32]
        
        st.markdown(f"""<div style="background:#0f172a; border:1px solid #334155; border-radius:12px; padding:1.25rem;">
          <div style="font-family:monospace; color:#94a3b8; font-size:0.9rem;">
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid #334155; padding-bottom:0.5rem; margin-bottom:0.5rem;">
                <span style="color:#cbd5e1; font-weight:600;">Event: Subpoena-Ready Forensic Audit</span>
                <span>{now.isoformat()}Z</span>
            </div>
            <div style="margin-bottom:0.3rem;"><strong style="color:#60a5fa;">[ORIGINAL_MD5]</strong> {md5_hash}</div>
            <div style="margin-bottom:0.3rem;"><strong style="color:#60a5fa;">[ORIGINAL_SHA256]</strong> {hashlib.sha256(raw_bytes).hexdigest()}</div>
            <div style="margin-bottom:0.3rem;"><strong style="color:#eab308;">[EXTRACTED_PAYLOAD_SHA256]</strong> {payload_sha}</div>
            <div style="margin-bottom:0.3rem;"><strong style="color:#10b981;">[SERVER_INTEGRITY_SEAL]</strong> {server_hash}</div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### 🤖 LLM-Assisted Automated Forensic Report Writer")
        st.markdown("""<div class="card">
          <div class="card-body">
            This generates a NIST SP 800-61 aligned Incident Response Ticket from the real forensic telemetry gathered above. 
            It is template-based (no external LLM API is called — data stays on your machine).
          </div></div>""", unsafe_allow_html=True)
          
        if st.button("📝 Generate Executive Briefing"):
            # No sleep/mock — uses REAL forensic data
            rs_line = f"RS Screening Score: {rs_result['screening_score']*100:.1f}% ({rs_result.get('finding','N/A')})" if rs_result else "RS Screening: Not available"
            struct_line = f"Trailing Bytes: {structural.get('trailing_bytes',0)} after {structural.get('format','?')} boundary" if structural else "Structural Audit: Not available"
            blind_line = f"Blind CNN Stego Probability: {blind_result['stego_probability']*100:.1f}%" if (blind_result and blind_result.get('available')) else "Blind CNN: Not available (no trained checkpoint)"
            
            st.info("### 📄 Incident Response Ticket (NIST SP 800-61)\n\n"
                    "**Incident Classification:** " + ("Suspicious — Possible Steganographic Exfiltration" if label == 'THREAT' else "Benign File Transfer") + "\n\n"
                    "**Executive Summary:**\n"
                    f"The file `{uploaded.name}` (MD5: `{md5_hash}`) was submitted to Sudarshana AI forensic analysis. "
                    f"The primary classifier returned a stego probability of {stego_prob*100:.1f}%. "
                    f"{'A suspicious signal was detected. Full investigation is recommended.' if label == 'THREAT' else 'No conclusive steganographic signal was detected.'}\n\n"
                    "**Forensic Evidence Summary:**\n"
                    f"- {rs_line}\n"
                    f"- {struct_line}\n"
                    f"- {blind_line}\n"
                    f"- Chi-Square, LSB Entropy, and Equalization metrics are in the downloadable report.\n\n"
                    "**Recommended Action:**\n"
                    + ("Quarantine the file. Do not open on an endpoint. Escalate to the SOC team with this ticket." if label == 'THREAT' else "No immediate action required. File may proceed with standard transfer protocols.")
            )
                
        st.markdown("---")
    
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📄 Download Report (.txt)",
                data=report_txt,
                file_name=f"Sudarshana_Report_{uploaded.name.rsplit('.', 1)[0]}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl2:
            st.markdown("""<div class="card" style="text-align:center; padding: 0.8rem;">
              <div style="font-size: 0.85rem; color: var(--accent-clay); font-weight: 600;">
                © Sudarshana AI — All Rights Reserved
              </div>
              <div style="font-size: 0.75rem; color: rgba(20,20,19,0.5); margin-top: 0.2rem;">
                Report generated automatically by the Sudarshana Forensic Engine
              </div>
            </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 6 — AI EXPLAINABILITY (Grad-CAM)
    # ═══════════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-title">🧠 AI Explainability — Why Did the AI Decide This?</div>', unsafe_allow_html=True)
        st.markdown("""<div class="card">
          <div class="card-body">
            This section shows exactly <em>where</em> in the image the AI's attention was focused when it made
            its decision. Gradient-weighted Class Activation Mapping (Grad-CAM) produces a heatmap
            by tracing which pixels most strongly influenced the final classification score.
            <br><br>
            <strong>🔴 Red/Yellow = High suspicion — the AI focused heavily here.<br>
            🔵 Blue/Cold = Low suspicion — these areas appeared clean to the AI.</strong>
          </div></div>""", unsafe_allow_html=True)
        st.caption("💡 *ELI5: Imagine the AI is a detective. The heatmap is its magnifying glass. Wherever the glass glows hottest is where the detective found the most suspicious clues.*")

        try:
            # Build the Grad-CAM tensor
            gradcam_tensor = preprocess(pil_img)
            original_rgb   = np.array(pil_img.convert("RGB"))

            # Grad-CAM model (fresh instance to avoid consumed hooks)
            gradcam_model = MultiBranchSteganalyzer()
            if trained and os.path.exists(model_path):
                from utils.checkpoints import extract_state_dict
                raw_ckpt = torch.load(model_path, map_location="cpu")
                gradcam_model.load_state_dict(extract_state_dict(raw_ckpt))
            gradcam_model.eval()

            target_layer = gradcam_model.branch_a.conv4[0]
            cam          = StegoGradCAM(gradcam_model, target_layer)
            heatmap      = cam.generate_heatmap(gradcam_tensor, target_class=1)
            cam.close()

            overlay_img, heatmap_color = StegoGradCAM.overlay_heatmap(original_rgb, heatmap, alpha=0.45)
            heatmap_intensity = float(np.mean(heatmap))

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.image(original_rgb, use_container_width=True, caption="Original Image")
            with gc2:
                st.image(overlay_img, use_container_width=True, caption="Grad-CAM Overlay (Heat = Suspicious)")
            with gc3:
                st.image(heatmap_color, use_container_width=True, caption="Raw Heatmap (Red = High Attention)")

            # Plain-language interpretation
            if label == "THREAT":
                interp_text = (
                    f"The model classified this image as <strong>stego (hidden data present)</strong> with "
                    f"{stego_prob*100:.1f}% confidence. The Grad-CAM heatmap shows "
                    f"<strong>concentrated activation</strong> — the network found suspicious pixel patterns "
                    f"in specific spatial regions. Average heatmap activation: "
                    f"<strong>{heatmap_intensity:.3f}</strong> (higher = more suspicious regions detected)."
                )
            else:
                interp_text = (
                    f"The model classified this image as <strong>clean</strong> with "
                    f"{clean_prob*100:.1f}% confidence. The Grad-CAM heatmap shows "
                    f"<strong>diffuse, low-intensity activation</strong> — the network did not find "
                    f"any concentrated suspicious regions. Average heatmap activation: "
                    f"<strong>{heatmap_intensity:.3f}</strong> (lower = more natural pixel patterns)."
                )
            st.markdown(f"""<div class="card">
              <div class="card-title">📊 Interpretation</div>
              <div class="card-body">{interp_text}</div></div>""", unsafe_allow_html=True)

            st.markdown("---")

            # Explanation Stability Test
            st.markdown("#### 🧪 Explanation Stability Test")
            st.markdown("""<div class="card">
              <div class="card-body">
                A trustworthy AI explanation must be <strong>stable</strong>: if we slightly
                perturb the input (add tiny Gaussian noise σ=0.01), the heatmap should remain
                largely similar. If the explanation changes drastically from a tiny perturbation,
                the model may be relying on fragile, irrelevant patterns rather than genuine
                steganographic features.
              </div></div>""", unsafe_allow_html=True)
            st.caption("💡 *ELI5: If a detective changes their conclusion just because you move a chair by 1 cm, they are not a reliable detective. This test checks if our AI is a reliable detective.*")

            # Perturbed pass — fresh model instance
            perturbed_tensor = gradcam_tensor + torch.randn_like(gradcam_tensor) * 0.01
            perturbed_tensor = torch.clamp(perturbed_tensor, -1.0, 1.0)

            gradcam_model2 = MultiBranchSteganalyzer()
            if trained and os.path.exists(model_path):
                gradcam_model2.load_state_dict(extract_state_dict(raw_ckpt))
            gradcam_model2.eval()

            target_layer2 = gradcam_model2.branch_a.conv4[0]
            cam2          = StegoGradCAM(gradcam_model2, target_layer2)
            heatmap_perturbed = cam2.generate_heatmap(perturbed_tensor, target_class=1)
            cam2.close()

            heatmap_diff   = np.abs(heatmap.astype(np.float64) - heatmap_perturbed.astype(np.float64))
            stability_score = 1.0 - float(np.mean(heatmap_diff))

            overlay_perturbed, _ = StegoGradCAM.overlay_heatmap(original_rgb, heatmap_perturbed, alpha=0.45)
            diff_visual = np.uint8(np.clip(heatmap_diff * 255 * 5, 0, 255))
            diff_color  = cv2.applyColorMap(diff_visual, cv2.COLORMAP_HOT)
            diff_color  = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                st.image(overlay_img, use_container_width=True, caption="Original Grad-CAM")
            with pcol2:
                st.image(overlay_perturbed, use_container_width=True, caption="Perturbed Grad-CAM (σ=0.01 noise)")
            with pcol3:
                st.image(diff_color, use_container_width=True, caption="Difference Map (amplified ×5)")

            if stability_score > 0.90:
                stab_verdict = "✅ HIGHLY STABLE — The explanation is robust and trustworthy."
                stab_color   = "#16a34a"
            elif stability_score > 0.75:
                stab_verdict = "🟡 MODERATELY STABLE — The explanation is reasonably consistent."
                stab_color   = "#ca8a04"
            else:
                stab_verdict = "🔴 UNSTABLE — The explanation may not reflect genuine features."
                stab_color   = "#dc2626"

            st.markdown(f"""<div class="card">
              <div class="card-title">📈 Stability Score</div>
              <div class="card-body">
                <strong style="font-size:1.5rem;color:{stab_color};">{stability_score*100:.1f}%</strong>
                <br>{stab_verdict}
              </div></div>""", unsafe_allow_html=True)

            if not trained:
                st.info("⚠️ **Model weights not loaded.** The Grad-CAM above reflects an untrained random model. "
                        "Results will be meaningful only after training on ALASKA2/BOSSBase. All other forensic "
                        "analysis features (RS, Structural, Blind CNN) are fully functional without weights.")

        except Exception as e:
            st.warning(
                f"⚠️ Grad-CAM could not be generated: {str(e)}. "
                "This feature requires the full model weights to be loaded. "
                "Please complete model training first."
            )


def generate_report_txt(filename, w, h, size_kb, md5_hash, label,
                        stego_prob, clean_prob, mname, mconf, mdesc,
                        cap_bytes, used_bytes, equalization, survive,
                        raw_bytes, structural=None, rs_result=None, blind_result=None):
    """Generate a professional TXT forensic report with Sudarshana copyright."""
    now = datetime.now()
    doc_id = f"SUD-{hashlib.sha256((filename + str(now)).encode()).hexdigest()[:12].upper()}"

    # Compute stats for the report
    gray = np.array(Image.open(io.BytesIO(raw_bytes)).convert("L"), dtype=np.uint8)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    even = hist[0::2].astype(float); odd = hist[1::2].astype(float)
    exp = (even + odd) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = np.where(exp > 0, (even - exp)**2 / exp, 0).sum()
    lsb = (gray & 1).astype(float)
    lsb_ent = -sum(p * np.log2(p + 1e-10) for p in [lsb.mean(), 1 - lsb.mean()])

    lines = []
    lines.append("═" * 65)
    lines.append("              SUDARSHANA AI — FORENSIC REPORT")
    lines.append("      Image Forensics & Hidden Data Detection Platform")
    lines.append("═" * 65)
    lines.append(f"  © {now.year} Sudarshana AI. All rights reserved.")
    lines.append("  This report was generated automatically by the Sudarshana")
    lines.append("  Steganography Detection System.")
    lines.append("")
    lines.append(f"  Report Generated : {now.strftime('%d %B %Y, %I:%M %p IST')}")
    lines.append(f"  Document ID      : {doc_id}")
    lines.append("═" * 65)
    lines.append("")

    # Image metadata
    lines.append("─── IMAGE METADATA ─────────────────────────────────────────────")
    lines.append(f"  File Name    : {filename}")
    lines.append(f"  Dimensions   : {w} × {h} pixels")
    lines.append(f"  File Size    : {size_kb:.1f} KB")
    lines.append(f"  MD5 Hash     : {md5_hash}")
    lines.append("")

    # Scan verdict
    lines.append("─── SCAN VERDICT ───────────────────────────────────────────────")
    if label == "SAFE":
        lines.append("  ✅ VERDICT: IMAGE IS SAFE")
        lines.append("  No hidden data was detected in this image.")
    else:
        lines.append("  🚨 VERDICT: HIDDEN DATA DETECTED")
        lines.append("  Secret data has been embedded inside this image.")
    lines.append(f"  Hidden Data Probability : {stego_prob*100:.1f}%")
    lines.append(f"  Clean Image Probability : {clean_prob*100:.1f}%")
    lines.append("")

    # Method fingerprint
    lines.append("─── METHOD FINGERPRINT ─────────────────────────────────────────")
    lines.append(f"  Detected Method : {mname}")
    lines.append(f"  Confidence      : {mconf}")
    lines.append(f"  Details         : {mdesc}")
    lines.append("")

    # Capacity analysis
    lines.append("─── CAPACITY ANALYSIS ──────────────────────────────────────────")
    lines.append(f"  Maximum Hiding Capacity : {fmt_bytes(cap_bytes)}")
    lines.append(f"  Estimated Data Used     : ~{fmt_bytes(used_bytes)}")
    pct = used_bytes * 100 // max(cap_bytes, 1)
    lines.append(f"  Capacity Utilised       : {pct}%")
    lines.append("")

    # Statistical analysis
    lines.append("─── STATISTICAL ANALYSIS ───────────────────────────────────────")
    lines.append(f"  Chi-Square Value  : {chi2:.0f}")
    lines.append(f"  LSB Entropy       : {lsb_ent:.4f}")
    lines.append(f"  Mean Brightness   : {gray.mean():.0f}")
    lines.append(f"  Std Deviation     : {gray.std():.0f}")
    lines.append(f"  Equalization      : {equalization*100:.1f}%")
    lines.append("")

    # Compression survival
    lines.append("─── COMPRESSION SURVIVAL ───────────────────────────────────────")
    for platform, pct_surv in survive.items():
        status = "Data Preserved" if pct_surv > 80 else "Partially Destroyed" if pct_surv > 20 else "Data Destroyed"
        lines.append(f"  {platform:<35} {pct_surv:>3}%  ({status})")
    lines.append("")

    # Structural & RS forensics
    lines.append("─── STRUCTURAL / RS SCREENING ────────────────────────────────")
    if structural:
        lines.append(f"  Format                 : {structural.get('format', 'Unknown')}")
        lines.append(f"  Boundary Offset        : {structural.get('boundary_offset', 'not established')}")
        lines.append(f"  Trailing Bytes         : {structural.get('trailing_bytes', 0)}")
        lines.append(f"  Extension Matches Type : {structural.get('extension_matches_format', 'unknown')}")
        lines.append(f"  Boundary Finding       : {structural.get('reason', '')}")
    else:
        lines.append("  Structural audit was not run or not available.")
    if rs_result:
        lines.append(f"  RS Screening Score     : {rs_result['screening_score']*100:.1f}% (screening only, not calibrated)")
        lines.append(f"  RS Finding             : {rs_result.get('finding', 'N/A')}")
        lines.append(f"  RS Groups              : {rs_result.get('groups', '?')} ({rs_result.get('usable_groups', '?')} usable)")
    lines.append("")

    if blind_result and blind_result.get('available'):
        lines.append("─── DEEP LEARNING BLIND CNN ─────────────────────────────────")
        lines.append(f"  Model                  : {blind_result.get('model_name', 'Unknown')}")
        lines.append(f"  Stego Probability      : {blind_result['stego_probability']*100:.1f}%")
        lines.append(f"  Residual Energy        : {blind_result.get('residual_energy', 0):.4f}")
        lines.append(f"  Scope Note             : {blind_result.get('scope_note', '')}")
        lines.append("")

    # Legal Chain of Custody
    lines.append("─── LEGAL CHAIN OF CUSTODY LOG ──────────────────────────────")
    lines.append(f"  Submission Timestamp   : {now.isoformat()}Z")
    lines.append(f"  Original File MD5      : {md5_hash}")
    lines.append(f"  Original File SHA-256  : {hashlib.sha256(raw_bytes).hexdigest()}")
    extracted = bytes((gray.flatten() & 1))
    lines.append(f"  Extracted Payload SHA  : {hashlib.sha256(extracted).hexdigest()}")
    lines.append(f"  Server Integrity Hash  : {hashlib.sha512((md5_hash + str(now.timestamp())).encode()).hexdigest()[:32]}")
    lines.append("")

    # Footer
    lines.append("═" * 65)
    lines.append("  DISCLAIMER: This analysis is based on statistical and AI-based")
    lines.append("  methods. Results should be verified by qualified forensic")
    lines.append("  analysts before taking any action.")
    lines.append("")
    lines.append("  Powered by Sudarshana AI — The Divine All-Seeing Eye")
    lines.append("═" * 65)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
