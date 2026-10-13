"""
api.py — FastAPI REST Backend for Sudarshana
=============================================
Provides a production-grade REST API for steganography detection.

This is designed for deployment behind DRDO/ISRO defense infrastructure,
Docker containers, or edge devices via NVIDIA Jetson.

ENDPOINTS:
  POST /analyze     → Upload an image for full steganalysis
  GET  /health      → Health check for container orchestrators
  GET  /model-info  → Model architecture and version information

USAGE:
  # Development
  uvicorn src.deploy.api:app --host 0.0.0.0 --port 8000 --reload

  # Production (Docker)
  uvicorn src.deploy.api:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys
import io
import hashlib
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.fusion_model import MultiBranchSteganalyzer
from utils.checkpoints import extract_state_dict


# ─── Pydantic Response Models ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str

class ModelInfoResponse(BaseModel):
    name: str
    version: str
    architecture: str
    branches: list
    fusion_type: str
    total_parameters: int
    model_size_mb: float
    quantized: bool

class AnalysisResult(BaseModel):
    filename: str
    verdict: str
    stego_probability: float
    clean_probability: float
    confidence: float
    detected_method: str
    method_confidence: str
    image_dimensions: str
    file_size_bytes: int
    md5_hash: str
    processing_time_ms: float
    timestamp: str


# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sudarshana API",
    description="Advanced Image Steganography Detection API — Powered by Multi-Branch CNN Fusion",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ───────────────────────────────────────────────────────────

DEVICE = torch.device("cpu")
MODEL = None
MODEL_TRAINED = False
MODEL_PATH = os.environ.get(
    "SUDARSHANA_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "weights", "fusion_best.pt")
)


def load_model():
    """Load the Sudarshana model at startup."""
    global MODEL, MODEL_TRAINED

    model = MultiBranchSteganalyzer()
    trained = False

    if os.path.exists(MODEL_PATH):
        try:
            raw = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(extract_state_dict(raw))
            trained = True
        except Exception as e:
            print(f"⚠️  Could not load weights: {e}")

    model.eval()

    # INT8 Dynamic Quantization for faster CPU inference
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    except RuntimeError:
        pass

    MODEL = model
    MODEL_TRAINED = trained
    print(f"✅ Model loaded (trained={trained}, device={DEVICE})")


@app.on_event("startup")
async def startup_event():
    load_model()


# ─── Image Preprocessing ─────────────────────────────────────────────────────

def preprocess(pil_img, sz=256):
    """Convert PIL image to model-ready tensor."""
    img = pil_img.convert("L")
    w, h = img.size
    if w < sz or h < sz:
        pw, ph = max(0, sz - w), max(0, sz - h)
        import torchvision.transforms.functional as TF
        img = TF.pad(img, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2))
    l = (img.width - sz) // 2
    t = (img.height - sz) // 2
    img = img.crop((l, t, l + sz, t + sz))
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)


def fingerprint_method(pil_img, sp):
    """Identify the likely steganographic method."""
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)
    lsb = (gray & 1).astype(float)
    lsb_mean = float(lsb.mean())
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 255))
    even = hist[0::2].astype(float)
    odd = hist[1::2].astype(float)
    expected = (even + odd) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(expected > 0, (even - expected) ** 2 / expected, 0).sum()

    if sp < 0.4:
        return "No Hidden Data Detected", "HIGH"
    elif abs(lsb_mean - 0.5) < 0.05:
        return "LSB Substitution Method", "HIGH"
    elif abs(lsb_mean - 0.5) < 0.15:
        return "LSB Matching (±1 Method)", "MEDIUM"
    elif chi < gray.size * 0.001:
        return "JPEG Frequency Method (DCT)", "MEDIUM"
    else:
        return "Adaptive / Unknown Method", "LOW"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for container orchestration (Kubernetes, Docker)."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "loading",
        model_loaded=MODEL_TRAINED,
        device=str(DEVICE),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model architecture and deployment information."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_params = sum(p.numel() for p in MODEL.parameters())
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024) if os.path.exists(MODEL_PATH) else 0

    return ModelInfoResponse(
        name="Sudarshana Multi-Branch Steganalyzer",
        version="2.0.0",
        architecture="3-Branch CNN Fusion (Pixel + DCT + Statistics)",
        branches=["Branch A (Pixel/SRM CNN)", "Branch B (DCT/Frequency CNN)", "Branch C (Statistics MLP)"],
        fusion_type="Dense (544→128→64→2)",
        total_parameters=total_params,
        model_size_mb=round(model_size, 2),
        quantized=True,
    )


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Upload an image for full steganography analysis.

    Accepts: JPEG, PNG, BMP, PGM
    Returns: Full forensic analysis result
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/x-portable-graymap"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: JPEG, PNG, BMP, PGM"
        )

    # Read and process the image
    start_time = time.time()
    raw_bytes = await file.read()
    md5_hash = hashlib.md5(raw_bytes).hexdigest()

    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        pil_img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    w, h = pil_img.size

    # Run inference
    tensor = preprocess(pil_img)
    with torch.no_grad():
        probs = F.softmax(MODEL(tensor), dim=1)
        stego_prob = float(probs[0, 1])
        clean_prob = float(probs[0, 0])

    verdict = "THREAT" if stego_prob >= 0.5 else "SAFE"
    confidence = max(stego_prob, clean_prob)
    method_name, method_conf = fingerprint_method(pil_img, stego_prob)

    processing_time = (time.time() - start_time) * 1000

    return AnalysisResult(
        filename=file.filename or "unknown",
        verdict=verdict,
        stego_probability=round(stego_prob, 4),
        clean_probability=round(clean_prob, 4),
        confidence=round(confidence, 4),
        detected_method=method_name,
        method_confidence=method_conf,
        image_dimensions=f"{w}×{h}",
        file_size_bytes=len(raw_bytes),
        md5_hash=md5_hash,
        processing_time_ms=round(processing_time, 1),
        timestamp=datetime.now().isoformat(),
    )
