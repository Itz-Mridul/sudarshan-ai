"""
demo.py — Sudarshana: Advanced Steganography Forensics Platform
Named after the divine Sudarshana Chakra of Lord Vishnu/Narayana.
White theme · Calibri font · Indian English
"""

import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
sys.path.insert(0, _SRC)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io, struct, base64, hashlib
from datetime import datetime
import pandas as pd

from model.fusion_model import MultiBranchSteganalyzer
from deploy.risk_scoring import compute_risk_score

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

def make_all_bitplanes(pil_img, channel="Grey"):
    ch_map = {"Grey": pil_img.convert("L"),
              "Red":  pil_img.convert("RGB"),
              "Green":pil_img.convert("RGB"),
              "Blue": pil_img.convert("RGB")}
    base = np.array(ch_map[channel], dtype=np.uint8)
    if channel == "Red":   base = base[:,:,0]
    elif channel == "Green": base = base[:,:,1]
    elif channel == "Blue":  base = base[:,:,2]
    return [Image.fromarray(get_bitplane(base, p).astype(np.uint8)) for p in range(8)]

def make_lsb_heatmap(pil_img, sensitivity=1):
    rgb  = np.array(pil_img.convert("RGB"), dtype=np.uint8)
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
    for sig, name, icon in sigs:
        if data.startswith(sig): return name, icon, sig.hex().upper()
    pr = sum(32<=b<127 for b in data[:64])
    if pr > 50: return "Plain Text Message", "📝", data[:4].hex().upper()
    return "Unknown / Encrypted Data", "🔒", data[:4].hex().upper()

def compression_survival(pil_img):
    is_jpeg = getattr(pil_img, 'format', '') == "JPEG"
    if is_jpeg:
        # JPEG re-compression may partially preserve DCT-domain stego
        return {
            "WhatsApp (auto-compresses)":        8,
            "Facebook (auto-compresses)":        12,
            "Instagram (auto-compresses)":       10,
            "Telegram (Send as File)":           95,
            "Email attachment (lossless)":       98,
            "Direct USB / file transfer":       100,
        }
    return {
        "WhatsApp (auto-compresses)":        2,
        "Facebook (auto-compresses)":         5,
        "Instagram (auto-compresses)":        3,
        "Telegram (Send as File)":           95,
        "Email attachment (lossless)":       98,
        "Direct USB / file transfer":       100,
    }

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
    survive                     = compression_survival(pil_img)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🛡️  Quick Result",
        "🗺️  Where is the Data?",
        "🔓  What was Hidden?",
        "🚀  Expert Forensics",
        "📥  Download Report"
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
        heatmap_img = make_lsb_heatmap(pil_img, sensitivity=sens)
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

        # Bit plane isolation
        st.markdown("#### 🔬 Bit-Layer Isolation — View Each Layer of the Image Separately")
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
        bitplanes = make_all_bitplanes(pil_img, channel)
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

        gray_arr  = np.array(pil_img.convert("L"), dtype=np.uint8)
        lsb_bits  = (gray_arr.flatten() & 1)[:2048]
        lsb_bytes = np.packbits(lsb_bits).tobytes()[:64]

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

        # Extraction attempt
        st.markdown("#### 🔑 Extraction with Password (If You Have the Key)")
        with st.expander("🔓 Attempt to extract hidden message using a password", expanded=False):
            st.markdown("""<div class="card">
              <div class="card-body">
                Some tools like Steghide and OpenStego protect the hidden data with a password.
                If you know the password that was used when embedding, kindly enter it below
                to attempt extraction. Without the correct password, the data will remain unreadable.
              </div></div>""", unsafe_allow_html=True)
            password = st.text_input("Kindly enter the password / key (leave blank if unprotected)", type="password")
            n_bytes  = st.slider("How many bytes to extract?", 8, 256, 64, 8)
            if st.button("🔓 Do the Needful — Attempt Extraction", use_container_width=True):
                extracted = lsb_bytes[:n_bytes]
                if password:
                    key = (password.encode() * (n_bytes // max(len(password),1) + 1))
                    extracted = bytes(b ^ k for b, k in zip(extracted, key[:n_bytes]))
                decoded_att = ''.join(chr(b) if 32<=b<127 else '·' for b in extracted)
                pr2 = sum(32<=b<127 for b in extracted)/max(len(extracted),1)
                st.markdown(f'<div class="hex-box">{decoded_att}</div>', unsafe_allow_html=True)
                if pr2 > 0.6:
                    st.success(f"✅ {pr2*100:.0f}% is readable text — kindly verify if the content above is meaningful!")
                else:
                    st.warning(f"🔒 {pr2*100:.0f}% is readable — wrong password, different tool was used, or no hidden data is present.")

        st.markdown("---")

        # File type detection
        st.markdown("#### 🏷️ Hidden File Type Detection — What Kind of Data is Hidden?")
        dtype, dicon, dmagic = get_magic_bytes(lsb_bytes)
        st.markdown(f"""<div class="card">
          <div class="card-title">🪄 Payload Signature Analysis</div>
          <div class="card-body">
            First bytes of the extracted LSB data:
            <code style="color:#1d4ed8;background:#eff6ff;padding:.1rem .5rem;
            border-radius:4px;font-family:monospace;">{dmagic}</code>
            <br><br>
            Detected type: {dicon} <strong>{dtype}</strong>
            <br><br>
            {'🚨 <strong style="color:#dc2626;">SERIOUS WARNING:</strong> Executable file signature detected! '
             'A computer programme may be hidden inside this image. '
             'This is a common malware distribution technique. '
             'Kindly do not open this file on any important system.'
             if 'Malware' in dtype or 'Executable' in dtype or 'Programme' in dtype else
             '🔒 Data appears encrypted or random. This means either the image is clean, '
             'or there is encrypted data that requires a password to read.'
             if 'Encrypted' in dtype or 'Unknown' in dtype else
             '📄 A recognisable file signature was detected in the hidden layer. '
             'This strongly suggests intentional data embedding.'}
          </div></div>""", unsafe_allow_html=True)


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

        # Compression tolerance
        st.markdown("#### 📱 Compression Survival Score — Will the Hidden Data Survive?")
        st.markdown("""<div class="card">
          <div class="card-body">
            When an image is uploaded to social media platforms, those platforms automatically
            re-compress the image to save server space. This re-compression destroys
            steganographic data hidden in the pixel layer (LSB method).
            Only lossless channels like Telegram (Send as File) or direct USB transfer
            will preserve the hidden data intact.
          </div></div>""", unsafe_allow_html=True)
        for platform, survival in survive.items():
            col_a, col_b = st.columns([2, 3])
            col_a.write(f"**{platform}**")
            bar_color = "#16a34a" if survival > 80 else "#f97316" if survival > 20 else "#dc2626"
            verdict   = "✅ Data Preserved" if survival > 80 else "⚠️ Partially Destroyed" if survival > 20 else "❌ Data Destroyed"
            col_b.markdown(f"""
            <div style="background:#f1f5f9;border-radius:8px;overflow:hidden;height:26px;display:flex;align-items:center;">
              <div style="width:{survival}%;background:{bar_color};height:100%;border-radius:8px;
                display:flex;align-items:center;justify-content:flex-end;padding-right:.4rem;
                font-size:.72rem;font-weight:800;color:#fff;min-width:2.5rem;">
                {survival}%
              </div>
            </div>
            <div style="font-size:.75rem;color:{bar_color};font-weight:600;margin-top:.15rem;">{verdict}</div>
            """, unsafe_allow_html=True)

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
    
        report_txt = generate_report_txt(
            filename=uploaded.name, w=w, h=h, size_kb=size_kb,
            md5_hash=md5_hash, label=label,
            stego_prob=stego_prob, clean_prob=clean_prob,
            mname=mname, mconf=mconf, mdesc=mdesc,
            cap_bytes=cap_bytes, used_bytes=used_bytes,
            equalization=equalization, survive=survive,
            raw_bytes=raw_bytes,
        )
    
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


def generate_report_txt(filename, w, h, size_kb, md5_hash, label,
                        stego_prob, clean_prob, mname, mconf, mdesc,
                        cap_bytes, used_bytes, equalization, survive,
                        raw_bytes):
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
