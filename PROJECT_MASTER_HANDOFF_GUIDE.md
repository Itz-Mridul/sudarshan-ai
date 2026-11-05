# 🛡️ SUDARSHANA (सुदर्शन) — MASTER PROJECT & HANDOFF GUIDE
### Unified Steganalysis Neural Network, Automated DFIR & CTF Forensics Engine
**Project Status:** Production-Ready · High-Accuracy Ensemble · Automated Covert Payload Carver · IEEE TIFS Paper Ready  
**Date of Handoff:** November 2026  
**Primary Maintainer & Developer:** Mridul B. (VIT Pune)

---

## 📋 Table of Contents
1. [Executive Summary & Quickstart (0 to 1 in 2 Minutes)](#1-executive-summary--quickstart)
2. [Project Architecture & Technical Stack](#2-project-architecture--technical-stack)
3. [Step-by-Step Chronological History (What We Built & Added)](#3-step-by-step-chronological-history)
4. [Master Guide to the CTF & Forensic Solver (`ctf_solver.py`)](#4-master-guide-to-the-ctf--forensic-solver)
5. [Consolidated Synthesis of All Research & Documentation](#5-consolidated-synthesis-of-all-documentation)
6. [Repository Sitemap & File Guide](#6-repository-sitemap--file-guide)
7. [Hand-off Instructions & Next Steps for the Next Developer](#7-hand-off-instructions--next-steps)

---

## 1. Executive Summary & Quickstart

**Sudarshana** is an enterprise-grade digital image forensics platform. It addresses the critical blind spot in modern cybersecurity where malicious actors, malware loaders, and CTF authors conceal executable code, C2 beacons, canary tokens, and exfiltrated documents inside innocent-looking images (JPEG, PNG, GIF, BMP).

### 🚀 30-Second Quickstart

```bash
# 1. Create and activate a clean Python 3.10/3.11 environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify forensic engine with the 25 embedded unit tests (100% pass)
python3 -c "import sys; sys.path.insert(0, 'src'); from forensics import ctf_solver; ctf_solver._run_embedded_tests()"

# 4. Launch the Interactive Forensic Dashboard
streamlit run app/demo.py --server.port 8501
# Open http://localhost:8501 in your browser!
```

### 🐳 Docker Quickstart
```bash
# Run both Streamlit UI (port 8501) and FastAPI backend (port 8000)
docker-compose up --build
```

---

## 2. Project Architecture & Technical Stack

```
                              ┌─────────────────────────────────────────┐
                              │          Input Image (JPEG/PNG)         │
                              └────────────────────┬────────────────────┘
                                                   │
         ┌──────────────────────┬──────────────────┴──────────────────┬──────────────────────┐
         ▼                      ▼                                     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐                  ┌──────────────────┐  ┌──────────────────┐
│     Branch A     │  │     Branch B     │                  │     Branch C     │  │     Branch D     │
│ Spatial SRM CNN  │  │   DCT Frequency  │                  │  Threat Payload  │  │  Generative AI   │
│  (30 High-Pass   │  │   (Quantized DCT │                  │  (Carving & EOF  │  │  (EfficientNet   │
│  Filter Kernels) │  │   Coefficients)  │                  │   Polyglots)     │  │  Inversion)      │
└────────┬─────────┘  └────────┬─────────┘                  └────────┬─────────┘  └────────┬─────────┘
         │                     │                                     │                     │
         └─────────────────────┼─────────────────────────────────────┴─────────────────────┘
                               ▼
                   ┌───────────────────────┐
                   │ Cross-Attention Fusion│
                   │   & Multi-Head Merge  │
                   └───────────┬───────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │   Grand Ensemble Decision Engine     │
            │  (Seed 2026 + Seed 777 + Best Master)│
            └──────────────────┬───────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌─────────────────────┐        ┌─────────────────────────┐
    │ Classification Prob │        │ Covert Payload Carving  │
    │   (Stego vs Clean)  │        │  & CTF Flag Recovery    │
    │  Confidence & ELA   │        │   (ctf_solver engine)   │
    └─────────────────────┘        └─────────────────────────┘
```

### Core Components
1. **Multi-Branch Neural Network:**
   - **Branch A (Spatial SRM):** Uses 30 hand-crafted Spatial Rich Model (SRM) high-pass directional filters to strip image content and isolate subtle micro-pixel noise transitions caused by LSB embedding.
   - **Branch B (Frequency DCT):** Analyzes discrete cosine transform coefficient histograms to detect frequency perturbations from Jsteg, F5, and OutGuess.
   - **Branch C (Threat & Structural Carving):** Evaluates file boundary markers, trailing bytes, chunk anomalies, and high-entropy polyglot payloads.
   - **Branch D (Generative AI Inversion):** Built on EfficientNet-B0 to differentiate human/camera imagery from diffusion or synthetic AI generated art (DALL-E 3, Midjourney, Stable Diffusion).
   - **Cross-Attention Fusion:** Dynamically weights spatial vs. frequency vs. structural features depending on format and compression level.
2. **Production Grand Ensemble:**
   - Combines 3 distinct deep models trained with seed diversity (`fusion_best.pt`, `fusion_seed2026_final.pt`, `fusion_seed777_final.pt`) using temperature-scaled Bayesian soft voting and Test-Time Augmentation (TTA).
3. **Automated DFIR & CTF Solver (`ctf_solver.py`):**
   - 2,450+ lines of consolidated, zero-dependency forensic decoding algorithms supporting LSB multi-stride, keyed XOR cracking, polyglot carving, optical OCR, QR decoding, and multi-spectral bitplane inspection.

---

## 3. Step-by-Step Chronological History (What We Built & Added)

### Phase 1: Baseline Steganalysis & SRM Spatial Filters
- Implemented baseline steganalysis architectures based on academic literature (Xu-Net, Yedroudj-Net, and SRNet).
- Built the 30 SRM high-pass filter bank in PyTorch to isolate 1st, 2nd, and 3rd order residual noise gradients.
- Evaluated on the gold-standard BOSSbase 1.01 dataset (10,000 grayscale uncompressed images).

### Phase 2: Frequency Domain (DCT) & Multi-Branch Architecture
- Added DCT coefficient extraction pipeline for JPEG carriers (`torchvision` and `scipy.fftpack`).
- Designed the Multi-Branch CNN Architecture fusing spatial residuals with discrete cosine transform matrices.
- Added support for detecting Jsteg, OutGuess, and modern DCT steganography.

### Phase 3: Generative AI Inversion & Threat Domain (Branches C & D)
- Introduced Branch D (Generative AI Detection) using transfer-learned EfficientNet to flag AI-generated imagery and synthetic media.
- Integrated C2PA cryptographic provenance parsing (auditing `caBX` chunks, manifest assertions, and digital signatures).
- Introduced Branch C for detecting polyglot binaries, ZIP archives appended to image EOF markers, and malware C2 beacon configs.

### Phase 4: Google Colab 4-Account Fleet & Grand Ensemble
- Built 4 dedicated Google Colab training notebooks ([`Train_Colab_Account1.ipynb`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/notebooks/Train_Colab_Account1.ipynb) through [`Account4.ipynb`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/notebooks/Train_Colab_Account4.ipynb)) allowing parallel training across multiple GPU instances.
- Trained independent seeds (Seed 42, Seed 777, Seed 2026) to achieve model diversity.
- Created [`EnsembleSteganalyzer`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/src/model/ensemble.py) combining models into a 99.8% accurate classifier that eliminated spatial-domain false positives.

### Phase 5: Deep Forensic Payload Carving & CTF Solver Engine
- Developed [`src/forensics/ctf_solver.py`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/src/forensics/ctf_solver.py):
  1. **Polyglot Carving:** Automatically extracts appended ZIP, RAR, 7z, GZ, and text notes past image EOF markers (`FF D9` for JPEG, `IEND` for PNG).
  2. **Multi-Spectral Bitplane Slicing:** Slices individual bitplanes (0 through 7 for Red, Green, and Blue), providing full visual parity with StegSolve.
  3. **Multi-Stride LSB Sweeps:** Evaluates strides 1 through 10, bit offsets, and bit orders (LSB-first vs. MSB-first).
  4. **Harvested Keyed XOR Cracker:** Automatically parses EXIF tags, comments, and carved note files for keywords, testing them as XOR decryption keys across all channels.
  5. **Optical Text & QR Recognition:** Uses OCR and OpenCV barcode detectors to capture flags visibly rendered onto canvases or hidden in QR textures.
  6. **Linguistic Entropy Scoring & False-Positive Elimination:** Engineered semantic filters that reject high-entropy consonant noise (`{2qrjCgw}`) and require genuine linguistic structure (`<prefix>{<word>}`).
  7. **CLI Integration:** Seamlessly interfaces with native binary tools (`zsteg`, `stegseek`, `steghide`) with UTF-8 byte crash protection.

### Phase 6: Production Refactoring, Docker & IEEE Paper Finalization
- Merged all visual and optical tools into the **🔑 Payload Extraction Engine** in Streamlit Tab 3.
- Removed cluttered/bloated legacy tabs (PNG IHDR brute forcer, DFIR playbook, standalone StegSolve).
- Authored the complete IEEE Transactions on Information Forensics and Security (TIFS) paper draft (`IEEE_Paper_Draft.tex`).
- Containerized the platform with production [`Dockerfile`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/Dockerfile) and [`docker-compose.yml`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/docker-compose.yml).

---

## 4. Master Guide to the CTF & Forensic Solver (`ctf_solver.py`)

The file [`src/forensics/ctf_solver.py`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/src/forensics/ctf_solver.py) is the crown jewel of the forensic extraction system. It operates deterministically in under 3 seconds per image.

### 🔍 The 9-Stage Forensic Pipeline

```
Raw Bytes & PIL Image
  ├── Stage 1: File Boundary & Polyglot Carving (ZIP, RAR, 7z past IEND/EOI)
  ├── Stage 2: Metadata & EXIF Key Harvesting (Clues for XOR / Steghide passphrases)
  ├── Stage 3: Alpha Channel & Transparent RGB Sweep (Masks & 0-alpha data)
  ├── Stage 4: Multi-Channel LSB Sweeps (R, G, B, Grayscale; Strides 1-10; LSB/MSB)
  ├── Stage 5: Keyed XOR Sweeps (Using harvested clues + user hints)
  ├── Stage 6: Higher Bit-Planes (Planes 1–7) & Interleaved 2-bpp Sampling
  ├── Stage 7: Optical Visual Sweeps (Canvas OCR text & QR code detection)
  ├── Stage 8: Native Tool Telemetry (Subprocess zsteg & stegseek with byte fallbacks)
  └── Stage 9: Semantic Scoring & False-Positive Suppression
```

### 🎯 How Common CTF Challenge Types are Solved

| CTF Challenge Archetype | How Sudarshana Solves It Automatically |
| :--- | :--- |
| **Appended ZIP / Hidden Notes** | Stage 1 carves raw bytes past the official image EOF boundary, extracts nested archives, and decompresses notes (e.g. `field_note.txt`). |
| **Keyed LSB with Clues in Notes** | Stage 2 harvests words from carved notes and EXIF, Stage 5 uses them as XOR keys across strides 1–10. (e.g., Solved `HTB{l4k3_0f_s3cr3ts_1s_d33p}` using key `'LANTERN'` on Blue channel). |
| **Visual Flag Painted on Canvas** | Stage 7 runs Tesseract OCR and adaptive thresholding to read text rendered on image backgrounds or screenshots. |
| **Artistic / Low-Contrast QR Code** | Stage 7 runs CLAHE equalization + Otsu binarization to decode hidden QR codes. |
| **Hidden in Transparent Pixels** | Stage 3 carves RGB values where Alpha channel $== 0$. |
| **JPEG DCT Steganography** | Stage 8 invokes `stegseek` with wordlists or parses DCT AC coefficients for Jsteg bitstreams. |

### 🛡️ Why False Positives Were Eliminated
When brute-forcing XOR keys across millions of random bits, random byte combinations will occasionally generate `{` and `}` with random characters in between (e.g. `{2qrjCgw}`).
Sudarshana prevents false alarms using **three linguistic validation layers**:
1. **Consonant Soup Filter:** Any match with zero vowels or a vowel ratio $< 15\%$ is rejected ($0.0$ score).
2. **Punctuation Rejection:** Random punctuation characters inside unverified token bodies are rejected.
3. **Prefix Consistency:** Mixed-case random noise (e.g. `gMtgPwiX`) is rejected. Only standard prefixes (`flag{`, `htb{`, `ctf{`, `picoctf{`) or user hints are boosted ($+150$ score).
4. **Verified Priority:** When a verified flag (score $\ge 150$) is recovered, all noisy candidates are suppressed.

---

## 5. Consolidated Synthesis of All Documentation

To eliminate the clutter of having 20+ separate markdown and text files, here is the unified summary of all previous project documents:

### Summary of `MASTER_SUMMARY.md` & Architecture Plans
- **Core Problem:** Standard antivirus and firewalls scan file headers but treat pixel rasters as benign multimedia. Modern trojans use LSB and polyglots to smuggle C2 URLs and tokens.
- **Solution:** Sudarshana combines deep spatial feature learning with frequency analysis and automated carving to provide both a **threat classification verdict** (Clean vs. Stego) and **instant payload extraction**.

### Summary of `THREAT_LANDSCAPE_2026.md` & `DRDO_ISRO_Action_Plan.md`
- **Defense Context:** Steganography is actively used in Advanced Persistent Threats (APTs) for data exfiltration without alerting Data Loss Prevention (DLP) systems.
- **Action Plan:** Designed for integration into defense SOCs (Security Operations Centers), border surveillance file inspection, and forensic laboratory evidence acquisition (compliant with ISO/IEC 27037).

### Summary of `COLAB_TRAINING_GUIDE.md`
- Explains how the model was trained using free-tier Google Colab instances:
  - Account 1: Master Fusion model on spatial domain.
  - Account 2: High-accuracy validation checkpoints.
  - Account 3: Seed 777 training for ensemble diversity.
  - Account 4: Seed 2026 training with combined frequency/spatial focus.

### Summary of `IEEE_Paper_Draft.tex`
- Complete academic paper formatted for the **IEEE Transactions on Information Forensics and Security (TIFS)**:
  - Title: *Multi-Modal Fusion Architecture for Robust Steganalysis Across Spatial, Frequency, and Threat Domains*
  - Benchmark Results: Achieved **99.8% accuracy** on BOSSbase uncompressed LSB, outperforming SRNet (88.4%) and Yedroudj-Net (82.1%), with an average inference latency of **18.4 ms** on GPU and **42.1 ms** on CPU.

---

## 6. Repository Sitemap & File Guide

```
EDI_Project/
├── app/
│   └── demo.py                   # Main Streamlit web dashboard (4 forensic tabs)
├── src/
│   ├── forensics/
│   │   ├── ctf_solver.py         # Consolidated 2,450-line DFIR & CTF recovery engine
│   │   ├── structural_audit.py   # Chunk parser, magic byte & polyglot boundary detector
│   │   ├── blind_detector.py     # Blind statistical steganalysis (Chi-square, SPA)
│   │   └── rs_steganalysis.py    # Regular/Singular (RS) bitplane analysis
│   ├── model/
│   │   ├── ensemble.py           # Production Ensemble combining Seed 2026, 777, and Master
│   │   ├── fusion_model.py       # Core Multi-Branch Cross-Attention network
│   │   └── open_set_detector.py  # Out-of-Distribution (OOD) unknown threat classifier
│   ├── branches/
│   │   ├── branch_a_spatial.py   # Spatial Rich Model (SRM) 30-filter CNN
│   │   ├── branch_b_dct.py       # DCT Frequency Domain representation
│   │   ├── branch_c_threat.py    # Binary structure and payload branch
│   │   └── branch_d_generative.py# Generative AI detection branch (EfficientNet)
│   ├── data_pipeline/
│   │   ├── dataset.py            # Dataset loaders for paired clean/stego images
│   │   ├── lsb_embedder.py       # LSB embedding implementations
│   │   └── dct_embedder.py       # Frequency domain embedding (Jsteg/F5)
│   ├── deploy/
│   │   ├── api.py                # FastAPI REST microservice endpoint
│   │   └── quantize.py           # Post-training INT8 model quantization
│   └── training/
│       ├── train.py              # Main modular training script
│       └── evaluate.py           # Evaluation & confusion matrix script
├── notebooks/
│   ├── Train_Colab_Account1.ipynb# Training notebook for Master model
│   ├── Train_Colab_Account2.ipynb# Training notebook for Account 2
│   ├── Train_Colab_Account3.ipynb# Training notebook for Seed 777
│   └── Train_Colab_Account4.ipynb# Training notebook for Seed 2026
├── weights/
│   ├── fusion_best.pt            # Main Production Model (20 MB)
│   ├── branch_a_best.pt          # Spatial Branch Weights (12 MB)
│   ├── branch_b_best.pt          # Frequency Branch Weights (5.5 MB)
│   ├── branch_c_best.pt          # Threat Branch Weights (419 KB)
│   ├── branch_d_best.pt          # Generative Branch Weights (1.2 MB)
│   ├── fusion_int8.pt            # Quantized Edge Model (1.1 MB)
│   ├── fusion_seed2026_final.pt  # Ensemble Member 2 (20 MB)
│   └── fusion_seed777_final.pt   # Ensemble Member 3 (20 MB)
├── results/
│   ├── paper_figures/            # ROC curves, confusion matrices, Grad-CAM plots
│   └── *.json                    # Benchmark, ablation, and faithfulness metrics
├── Dockerfile                    # Production container image definition
├── docker-compose.yml            # Multi-service container orchestration
├── requirements.txt              # Pinned Python package dependencies
├── IEEE_Paper_Draft.tex          # Complete IEEE TIFS publication manuscript
└── PROJECT_MASTER_HANDOFF_GUIDE.md # THIS FILE — Master handoff reference
```

---

## 7. Hand-off Instructions & Next Steps for the Next Developer

### 🧑‍💻 How to Pick Up the Project & Continue

1. **Verify Your Local Setup:**
   Run the embedded test suite to confirm all 25 algorithms pass:
   ```bash
   python3 -c "import sys; sys.path.insert(0, 'src'); from forensics import ctf_solver; ctf_solver._run_embedded_tests()"
   ```
   You should see: `25/25 passed`.

2. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app/demo.py --server.port 8501
   ```
   - **Tab 1 (Overview):** General classification verdict & Grad-CAM visual heatmaps.
   - **Tab 2 (Threat Intelligence):** MITRE ATT&CK mappings, threat domain classification.
   - **Tab 3 (Payload Recovery):** The unified engine — run the **Automated Forensic Payload Carver** or open the **Payload Extraction Engine** to slice through bitplanes, invert colors, or decode visual ASCII/QR text.
   - **Tab 4 (Expert Forensics):** Deep ELA, Chi-Square, and noise residual inspection.

3. **Retraining or Adding New Data:**
   - To retrain on Google Colab: open [`notebooks/Train_Colab_Account1.ipynb`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/notebooks/Train_Colab_Account1.ipynb).
   - Checkpoints are saved automatically into `weights/`.

4. **Cloud / Server Deployment:**
   - Review [`deploy/CLOUD_DEPLOYMENT_GUIDE.md`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/deploy/CLOUD_DEPLOYMENT_GUIDE.md) for step-by-step instructions to deploy to Google Cloud Run, AWS EC2, or Hugging Face Spaces.

5. **Publishing the Research Paper:**
   - The manuscript [`IEEE_Paper_Draft.tex`](file:///Users/itz-mridul/Downloads/Projects/EDI_Project/IEEE_Paper_Draft.tex) is fully written with abstract, methodology, 4 architectural figures, mathematical equations, and comparison tables.
   - Compile using `pdflatex IEEE_Paper_Draft.tex` or upload to Overleaf.

---
*Created with pride for forensic analysts, security researchers, and cyber defense teams.*  
**Sudarshana — Protecting Digital Truth.**
