# ─────────────────────────────────────────────────────────────────────────────
# Sudarshana — Unified Production Docker Image
# Multi-Branch CNN Steganalysis + Automated CTF Forensics Engine
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE}

LABEL maintainer="Sudarshana Forensics — VIT Pune"
LABEL description="AI Steganalysis + Automated CTF Solver (zsteg, steghide, stegseek, binwalk)"
LABEL version="2.5.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    SUDARSHANA_MODEL_PATH=/app/weights/fusion_best.pt \
    PORT=8501

WORKDIR /app

# 1. Install system libraries, Ruby (for zsteg), steghide, exiftool, binwalk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    ruby-full \
    rubygems \
    steghide \
    binwalk \
    libimage-exiftool-perl \
    p7zip-full \
    ca-certificates \
    && gem install zsteg --no-document \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Install StegSeek (High-speed rockyou cracker for JPEGs) on amd64
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "amd64" ]; then \
        wget -q https://github.com/RickdeJager/stegseek/releases/download/v0.6/stegseek_0.6-1.deb -O /tmp/stegseek.deb && \
        dpkg -i /tmp/stegseek.deb || apt-get install -fy && \
        rm -f /tmp/stegseek.deb; \
    fi

# 3. Provision standard CTF wordlist
RUN mkdir -p /usr/share/wordlists && \
    curl -sSL https://raw.githubusercontent.com/danielmiessler/SecLists/master/Passwords/Common-Credentials/10-million-password-list-top-10000.txt \
    -o /usr/share/wordlists/rockyou.txt

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy project source code and weights
COPY . .

# 6. Expose ports (8501 for Streamlit UI, 8000 for FastAPI)
EXPOSE 8501 8000

# 7. Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/health || exit 1

# 8. Default: Launch the Sudarshana Interactive Forensic & CTF Web UI
CMD ["streamlit", "run", "app/demo.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
