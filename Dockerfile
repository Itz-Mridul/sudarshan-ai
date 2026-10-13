# ─────────────────────────────────────────────────────────────────────────────
# Sudarshana — Docker Production Image
# Multi-Branch CNN Steganography Detection System
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   docker build -t sudarshana:latest .
#   docker run -p 8000:8000 sudarshana:latest
#
# For NVIDIA Jetson (edge deployment):
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3 -t sudarshana:jetson .
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Metadata
LABEL maintainer="Sudarshana AI Team — VIT Pune"
LABEL description="Steganography Detection API — Multi-Branch CNN Fusion"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SUDARSHANA_MODEL_PATH=/app/weights/fusion_best.pt

# System dependencies (OpenCV requires libgl1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Copy project files
COPY src/ ./src/
COPY weights/ ./weights/
COPY app/ ./app/

# Expose ports
# 8000 = FastAPI REST API
# 8501 = Streamlit Demo UI
EXPOSE 8000 8501

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run the FastAPI server
CMD ["uvicorn", "src.deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
