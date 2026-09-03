"""
demo.py
=======
Streamlit Web Demo — Steganography Detector (Step 13 / Checklist)

HOW TO RUN (from project root):
  streamlit run app/demo.py

WHAT THE USER SEES:
  1. Upload any image (JPG / PNG / PGM)
  2. The model runs all three branches + fusion
  3. Shows: CLEAN or STEGO + confidence %
  4. Shows: Defense Context Risk Score (with metadata inputs)

DEPENDENCIES:
  pip install streamlit pillow torch numpy
"""

import sys
import os

# Add project src to path (works regardless of where streamlit is launched from)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.join(_HERE, "..", "src")
sys.path.insert(0, _SRC)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model.fusion_model import MultiBranchSteganalyzer
from deploy.risk_scoring import compute_risk_score

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="StegShield — Steganography Detector",
    page_icon="🛡️",
    layout="wide",
)

# ─── Load Model (cached so it doesn't reload on every interaction) ─────────────

@st.cache_resource
def load_model(model_path: str = "weights/fusion_best.pt"):
    """Load and cache the trained fusion model."""
    device = torch.device("cpu")   # demo runs on CPU
    model = MultiBranchSteganalyzer()
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        st.success(f"✅ Model loaded from {model_path}")
    else:
        st.warning(
            f"⚠️ No trained model found at `{model_path}`. "
            "Using random weights for demo. Train the model first:\n\n"
            "`python src/training/train.py --mode fusion`"
        )
    model.eval()
    return model


def preprocess_image(uploaded_file, image_size: int = 256) -> tuple:
    """
    Convert an uploaded PIL image to a normalised tensor.

    Returns:
        (tensor, pil_img):
            tensor  — shape (1, 1, H, W) — batch of 1 grayscale image
            pil_img — PIL image for display
    """
    img = Image.open(uploaded_file).convert("L")      # convert to grayscale
    pil_display = img.copy()
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0     # normalise to [0, 1]
    # Normalise to [-1, 1] (matches training transforms)
    arr = (arr - 0.5) / 0.5
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor, pil_display


@torch.no_grad()
def predict(model: MultiBranchSteganalyzer, image_tensor: torch.Tensor) -> tuple:
    """
    Run inference and return (label, stego_probability).

    Returns:
        label       : "CLEAN" or "STEGO DETECTED"
        stego_prob  : float 0–1, probability of stego
    """
    logits = model(image_tensor)                        # (1, 2)
    probs  = F.softmax(logits, dim=1)                   # (1, 2)
    stego_prob = float(probs[0, 1])                     # P(stego)
    label = "STEGO DETECTED" if stego_prob >= 0.5 else "CLEAN"
    return label, stego_prob


# ─── UI Layout ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.title("🛡️ StegShield — Multi-Branch CNN Steganography Detector")
    st.markdown(
        "**DRDO Insider Threat Detection** | "
        "Pixel _(Branch A)_ + DCT _(Branch B)_ + Statistical _(Branch C)_ fusion | "
        "CPU-Only INT8 Quantized Deployment"
    )
    st.divider()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **MBCSS** — Multi-Branch CNN Steganalysis System

        **Architecture:**
        - Branch A: Pixel CNN + SRM filters (detects LSB)
        - Branch B: DCT frequency CNN (detects JPEG stego)
        - Branch C: Statistical MLP (chi-square, histogram)
        - Fusion: LayerNorm → Concat → Dense classifier

        **Dataset:** BOSSBase v1.01 (10,000 images)

        **Target:** 85-90% accuracy on mixed stego

        **Innovation:** First system combining all three domains
        with CPU-only INT8-quantized deployment + Defense context scoring
        """)
        st.divider()
        st.markdown("*2nd Year CSE | 3-Month Build*")
        st.markdown("*Stack: PyTorch + Streamlit*")
        model_path = st.text_input("Model path", value="weights/fusion_best.pt")

    # Two-column layout
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("📁 Upload Image")
        uploaded = st.file_uploader(
            "Upload any JPG / PNG / PGM image",
            type=["jpg", "jpeg", "png", "pgm"],
            help="The model will classify it as CLEAN or STEGO"
        )

        if uploaded:
            model = load_model(model_path)
            img_tensor, pil_img = preprocess_image(uploaded)
            st.image(pil_img, caption="Uploaded image (grayscale preview)", use_container_width=True)

            # ── Inference ──────────────────────────────────────────────────
            with st.spinner("Running all three branches + fusion..."):
                label, stego_prob = predict(model, img_tensor)
            clean_prob = 1.0 - stego_prob

            with col_result:
                st.subheader("🔍 Detection Result")

                # Verdict box
                if label == "STEGO DETECTED":
                    st.error(f"### 🚨 {label}")
                else:
                    st.success(f"### ✅ {label}")

                # Probability bars
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("P(Stego)", f"{stego_prob * 100:.1f}%")
                with c2:
                    st.metric("P(Clean)", f"{clean_prob * 100:.1f}%")

                st.progress(stego_prob, text=f"Stego probability: {stego_prob:.3f}")

                st.divider()

                # ── Defense Context Scoring (Patent Claim 2) ──────────────
                st.subheader("🔒 Defense Context Risk Score")
                st.caption(
                    "Fill in transfer metadata for a composite risk score. "
                    "Combines model output with 4 contextual factors. _(Patent Claim 2)_"
                )

                with st.form("risk_form"):
                    r1, r2 = st.columns(2)
                    with r1:
                        user_priv = st.selectbox(
                            "User Privilege",
                            ["intern", "contract", "employee", "manager", "admin"],
                            index=2
                        )
                        dest_type = st.selectbox(
                            "Destination Network",
                            ["internal", "dmz", "partner", "external"],
                            index=0
                        )
                    with r2:
                        transfer_hour = st.slider("Transfer Hour (24h clock)", 0, 23, 10)
                        file_size     = st.number_input("Actual File Size (MB)", value=1.0, min_value=0.01, step=0.1)
                        expected_size = st.number_input("Expected File Size (MB)", value=1.0, min_value=0.01, step=0.1)

                    user_id  = st.text_input("User ID (for report)", value="USR-????")
                    submitted = st.form_submit_button("🧮 Calculate Risk Score", use_container_width=True)

                if submitted:
                    metadata = {
                        "user_privilege":   user_priv,
                        "hour":             transfer_hour,
                        "file_size_mb":     file_size,
                        "expected_size_mb": expected_size,
                        "destination_type": dest_type,
                        "user_id":          user_id,
                        "filename":         uploaded.name,
                        "timestamp":        f"{transfer_hour:02d}:xx",
                    }
                    result = compute_risk_score(stego_prob, metadata)

                    # Display risk result with appropriate color
                    level = result["level"]
                    level_fn = {
                        "CRITICAL": st.error,
                        "HIGH":     st.warning,
                        "MEDIUM":   st.info,
                        "LOW":      st.success,
                    }
                    level_fn[level](
                        f"**Risk Score: {result['score']}/100 — {level}**\n\n"
                        f"{result['report']}"
                    )

                    with st.expander("📊 Score Breakdown by Factor"):
                        cols = st.columns(len(result["breakdown"]))
                        for i, (factor, score) in enumerate(result["breakdown"].items()):
                            cols[i].metric(factor.replace("_", " ").title(), f"+{score:.1f}")

    if not uploaded:
        with col_result:
            st.info("👈 Upload an image to begin analysis.")
            st.markdown("""
            **Expected flow:**
            1. Upload a JPG/PNG/PGM image
            2. All three branches run in parallel
            3. Fusion layer combines features
            4. See CLEAN/STEGO with confidence
            5. Fill metadata for Risk Score (0-100)
            """)

    # Footer
    st.divider()
    st.caption(
        "Multi-Branch CNN Steganalysis System (MBCSS) | 2nd Year CSE | "
        "Stack: PyTorch + Streamlit | "
        "Target journal: IETE Journal of Research (SCOPUS Q2) / Defence Science Journal (DRDO)"
    )


if __name__ == "__main__":
    main()
