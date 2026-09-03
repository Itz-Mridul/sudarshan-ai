# STEGANALYSIS PROJECT — MASTER RESEARCH DOCUMENT
### Multi-Branch CNN for Steganography Detection
**2nd Year CSE | 3-Month Build | Updated: Sep 2026**

---

## TABLE OF CONTENTS
1. [PROJECT MEMORY — Key Facts to Never Forget](#memory)
2. [Problem Statement & Research Gap](#problem)
3. [Existing Work — Papers, GitHub, Datasets](#existing-work)
4. [Proposed Architecture (Full)](#architecture)
5. [PRD — Product Requirements Document](#prd)
6. [Design Decisions & Rules](#rules)
7. [Implementation Checklist (Week-by-Week)](#checklist)
8. [Patent + Publication Strategy](#patent)
9. [Risk Register](#risks)

---

<a name="memory"></a>
## 1. PROJECT MEMORY — Key Facts to Never Forget

```
PROJECT NAME:  Multi-Branch CNN Steganalysis System (MBCSS)
GOAL:          Detect image steganography across LSB, DCT, and statistical domains
INNOVATION:    Pixel + DCT + Stats fusion  +  Defense-context scoring  +  INT8 CPU deploy
DATASET:       BOSSBase v1.01 (10,000 images, 512×512 grayscale, dde.binghamton.edu)
BASELINE:      Xu-Net (pixel-only), Yedroudj-Net (spatial, fails post-compression)
TARGET:        85-90% accuracy on mixed stego, <5% drop after INT8 quantization
STACK:         Python 3.10, PyTorch 2.x, OpenCV, scipy, scikit-learn, Streamlit
TRAINING HW:   Google Colab free T4 GPU
DEPLOY HW:     CPU-only laptop (air-gapped, no GPU)
PATENT:        File BEFORE any public disclosure (India absolute novelty rule)
JOURNALS:      IETE Journal of Research (SCOPUS), Defence Science Journal (DRDO)
LEGAL REF:     Patents Act 1970 + CRI Guidelines 2025 (effective July 29, 2025)
```

### Key Terminologies (Quick Ref)
| Term | Meaning |
|------|---------|
| LSB | Least Significant Bit — hides 1 bit per pixel, undetectable to eye |
| DCT | Discrete Cosine Transform — JPEG frequency domain hiding |
| SRM | Spatial Rich Model — 30 high-pass filters to amplify stego noise |
| bpp | Bits per pixel — payload density of hiding (0.4 bpp = standard test) |
| BOSSBase | Break Our Steganographic System — 10K image benchmark |
| PE | Probability of Error = ½(P_MD + P_FA) — standard stego metric |
| INT8 | 8-bit integer quantization — 4× size reduction, 1–5% accuracy loss |
| Air-gapped | Isolated network with no internet and no GPU — DRDO scenario |

---

<a name="problem"></a>
## 2. PROBLEM STATEMENT & RESEARCH GAP

### 2.1 The Real-World Threat
An insider at DRDO photographs a missile blueprint, embeds the PDF inside
a cricket-match JPEG using free steganography software (e.g., OpenStego,
Steghide), and emails it out. The image looks identical. No antivirus flags it.
No firewall catches it. This is **steganography-based data exfiltration**.

**How LSB works:**
```
Original pixel  = 200 (binary: 11001000)
Modified pixel  = 201 (binary: 11001001)  ← 1 bit changed
1 bit × millions of pixels = entire document hidden
Human eye: sees ZERO difference
```

### 2.2 Why Existing Tools Fail

| Tool | Failure Mode | Specific Limitation |
|------|-------------|---------------------|
| **Xu-Net** (Xu et al., 2016, IEEE Signal Processing Letters) | Pixel-only analysis | Fooled by JPEG/DCT-domain hiding; only catches spatial LSB |
| **Yedroudj-Net** (Yedroudj et al., 2018, ICASSP) | Spatial only, brittle | Accuracy degrades after JPEG compression or image resize |
| **Qian-Net / GNCNN** (Qian et al., 2015) | Weak at low payload | PE rises sharply at 0.1 bpp; limited to single algorithm |
| **SRNet** (Boroumand et al., 2019) | GPU-dependent | ~89% on S-UNIWARD but requires Tesla V-100 32GB GPU |
| **Antivirus / Firewall** | Not designed for stego | No pixel-level analysis; completely blind to LSB/DCT hiding |

### 2.3 Confirmed Research Gap (from ACM Computing Surveys 2024/25 + Frontiers AI 2025)
> "Most detectors are built for one specific hiding method and perform poorly
> against others. Generalisation across embedding algorithms remains the
> biggest unresolved problem in steganalysis."
> — ACM Computing Surveys 2024/25 review (400+ references)

**Your specific gap (refined, citable):**
> No existing published system combines: (a) pixel + DCT + statistical fusion,
> (b) CPU-only INT8-quantized deployment, AND (c) actionable defense-context
> risk scoring — as one integrated, air-gapped-deployable steganalysis pipeline
> for defense-sensitive environments.

---

<a name="existing-work"></a>
## 3. EXISTING WORK — Papers, GitHub, Datasets

### 3.1 Foundational Papers to Study (Chronological)

| Year | Paper | Key Contribution | Access |
|------|-------|-----------------|--------|
| 2012 | Fridrich & Kodovský — "Rich Models for Steganalysis" | SRM filters (30 high-pass kernels), ensemble classifier | IEEE TIFS |
| 2015 | Qian et al. — "GNCNN" | First competitive CNN for steganalysis | IEEE ICIP |
| 2016 | **Xu et al. — "Xu-Net"** | ABS layer + 1×1 kernels; YOUR PRIMARY BASELINE | IEEE Signal Processing Letters |
| 2018 | **Yedroudj et al. — "Yedroudj-Net"** | 7-block CNN with average-pooling; YOUR SECONDARY BASELINE | ICASSP |
| 2019 | Boroumand et al. — "SRNet" | Deep residual network; 89.77% on S-UNIWARD 0.4bpp | IEEE TIFS |
| 2022 | Zhang et al. — "Depth-wise separable CNN" | Efficient steganalysis, smaller model | IEEE TIFS |
| 2024 | Li et al. — "Lightweight + Residual + Transformer" | Multi-residual structure + Transformer fusion | Chinese J. Electronics |
| 2025 | Uspenskyi & Bondarchuk — "SRM + SSL for limited resources" | Self-supervised + SRM for CPU-scale deployment | IT&S journal |
| 2025 | Frontiers AI — Robustness study | Models degrade under compression/resize — backs your gap | Frontiers in AI |
| 2025 | **PENet+** — Kim et al. | Lightweight Transformer, 45.5% fewer params, edge focus | arXiv:2606.10939 |

**Critical papers to download and read first:**
1. Xu-Net: `G. Xu, H.-Z. Wu, and Y.-Q. Shi, IEEE Signal Processing Letters, vol. 23, no. 5, pp. 708–712, 2016.`
2. Yedroudj-Net: `M. Yedroudj, F. Comby, M. Chaumont, ICASSP 2018.`
3. SRNet: `M. Boroumand et al., IEEE TIFS, 2019.`
4. Rich Models (SRM): `J. Fridrich, J. Kodovsky, IEEE TIFS, 2012.`

### 3.2 GitHub Repositories to Leverage (Don't Start from Scratch)

| Repo | What to Reuse | URL |
|------|-------------|-----|
| **Xu-Net PyTorch** | Your baseline; reuse training loop + SRM filter init | github.com/rcouturier/steganalysis_with_CNN_and_SRM |
| **Yedroudj-Net (Caffe/PyTorch)** | Architecture reference for Branch A conv blocks | github.com/agorentala/steganalysis_with_CNN_Yedroudj-Net |
| **SRNet PyTorch** | Residual blocks for Branch A; pretrained weights available | github.com/yiwliu/Pytorch-implementation-of-SRNet |
| **SRNet TF2** | Paired batch generator — useful training technique | github.com/davidggz/SRNet-Tensorflow-Implementation |
| **Image-Steganalysis (ALASKA2)** | EfficientNet transfer learning approach | github.com/RishiMalhotra920/Image-Steganalysis |
| **LSB + DCT Embedder** | Ready-to-use LSB & DCT steganography in Python | github.com/Python-World/python-mini-projects/tree/master/projects/steganography |
| **Dilated Yedroudj-Net** | Dilated conv variant — good Branch A inspiration | github.com/kevin-planolles/steganalysis_with_cnn_dilated-yedroudj-net |
| **steganalysis (GitHub topics)** | Aggregated steganalysis repos | github.com/topics/steganalysis |

**How to use existing code without plagiarizing:**
- Use SRM filter weights as initialization (this is standard practice; cite Fridrich 2012)
- Use existing training loop structure but rewrite for your 3-branch architecture
- Cite every repo you adapt in your report's "Implementation" section

### 3.3 Datasets

| Dataset | Size | Format | Source | Use in Your Project |
|---------|------|--------|--------|-------------------|
| **BOSSBase v1.01** | 10,000 images, 512×512 | Grayscale PGM | dde.binghamton.edu/download/ | PRIMARY — create stego pairs from this |
| **BOWS2** | 10,000 images, 512×512 | Grayscale | http://bows2.ec-lille.fr | Secondary baseline extension |
| **DIV2K** | 800 high-res color images | RGB PNG | data.vision.ee.ethz.ch/cvl/DIV2K/ | Generalisation testing (color) |
| **ALASKA2** | 25,000 images | Grayscale/Color | Kaggle | Advanced testing; DCT-domain hiding |

**Dataset split (use this exact protocol — standard in literature):**
```
BOSSBase v1.01 (10,000 cover images)
├── Training set:   7,000 cover + 7,000 stego  (70%)
├── Validation set: 1,000 cover + 1,000 stego  (10%)
└── Test set:       2,000 cover + 2,000 stego  (20%)
```

**Creating stego images (Week 2 task):**
```python
# Simple LSB embedder (~20 lines)
from PIL import Image
import numpy as np

def embed_lsb(cover_path, payload_bits, output_path):
    img = Image.open(cover_path).convert('L')  # grayscale
    pixels = np.array(img)
    flat = pixels.flatten()
    for i, bit in enumerate(payload_bits):
        flat[i] = (flat[i] & 0xFE) | bit  # zero last bit, set new
    stego = Image.fromarray(flat.reshape(pixels.shape))
    stego.save(output_path)
    return stego
```

### 3.4 Steganography Algorithms to Test Against (Not Just LSB)

| Algorithm | Domain | Difficulty | Implementation Source |
|-----------|--------|-----------|----------------------|
| **LSB** (simple) | Spatial | Easy to detect | Write yourself (~20 lines) |
| **WOW** | Spatial adaptive | Medium | dde.binghamton.edu (MATLAB + Python wrapper) |
| **S-UNIWARD** | Spatial + JPEG | Hard | dde.binghamton.edu |
| **HILL** | Spatial adaptive | Hard | dde.binghamton.edu |
| **J-UNIWARD** | JPEG DCT domain | Hard | dde.binghamton.edu |

**Minimum viable test set for your paper:** LSB + S-UNIWARD + J-UNIWARD
(covers spatial, adaptive spatial, and JPEG frequency domains)

---

<a name="architecture"></a>
## 4. PROPOSED ARCHITECTURE (FULL DETAIL)

### 4.1 System Overview

```
INPUT IMAGE (any JPG/PNG)
        │
        ├─────────────────┬──────────────────┐
        ▼                 ▼                  ▼
   [BRANCH A]        [BRANCH B]          [BRANCH C]
  Pixel CNN         DCT CNN            Stats MLP
  (spatial)         (frequency)        (statistical)
        │                 │                  │
        ▼                 ▼                  ▼
  feature_a [256]   feature_b [256]   feature_c [64]
        │                 │                  │
        └─────────────────┴──────────────────┘
                          │
                  [CONCAT + NORMALIZE]
                  vector [576]
                          │
                 [FUSION LAYER]
                 Dense(128) → ReLU → Dropout(0.5)
                 Dense(64)  → ReLU → Dropout(0.3)
                 Dense(2)   → Softmax
                          │
              ┌───────────┴──────────────┐
           P(CLEAN)              P(STEGO) + confidence %
                                          │
                               [DEFENSE CONTEXT MODULE]
                               + user privilege score
                               + time anomaly score
                               + file size anomaly score
                               + destination IP score
                                          │
                                [RISK SCORE: 0-100]
                               LOW / MEDIUM / HIGH / CRITICAL
```

### 4.2 Branch A — Pixel Residual CNN

**Purpose:** Detect spatial domain hiding (LSB, WOW, HILL)

```python
class BranchA(nn.Module):
    def __init__(self):
        super().__init__()
        # SRM preprocessing (30 high-pass filters, initialized from Fridrich 2012)
        self.srm = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        # SRM weights are FIXED (not trained) — loaded from precomputed values
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(30, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(128 * 4 * 4, 256)
        self.abs_act = lambda x: torch.abs(x)  # ABS activation (from Xu-Net)
    
    def forward(self, x):
        x = self.abs_act(self.srm(x))       # pixel residuals amplified
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)                   # → 256-dim feature vector
```

**SRM Filter Initialization:**
```python
# Load precomputed SRM kernels (30 filters × 5×5)
# Available at: http://dde.binghamton.edu/download/
# Or generate: scipy.io.loadmat('srm_filters.mat')['F']
srm_weights = load_srm_filters()  # shape: (30, 1, 5, 5)
branch_a.srm.weight.data = torch.tensor(srm_weights)
branch_a.srm.weight.requires_grad = False  # freeze SRM filters
```

**Expected accuracy (solo):** 82–87% on LSB stego at 0.4 bpp

---

### 4.3 Branch B — DCT Frequency CNN

**Purpose:** Detect JPEG-domain / frequency hiding (J-UNIWARD, UERD)

```python
import scipy.fft as fft

def compute_dct_blocks(image_array, block_size=8):
    """Compute DCT on 8x8 blocks (same as JPEG)"""
    H, W = image_array.shape
    dct_map = np.zeros_like(image_array, dtype=np.float32)
    for i in range(0, H - block_size + 1, block_size):
        for j in range(0, W - block_size + 1, block_size):
            block = image_array[i:i+block_size, j:j+block_size].astype(float)
            dct_map[i:i+block_size, j:j+block_size] = fft.dct(
                fft.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho'
            )
    return dct_map

class BranchB(nn.Module):
    def __init__(self):
        super().__init__()
        # DCT coefficient map input (1 channel, same size as image)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.gpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(128 * 4 * 4, 256)
    
    def forward(self, x):  # x = DCT coefficient map
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gpool(x)
        x = x.flatten(1)
        return self.fc(x)                   # → 256-dim feature vector
```

**Expected accuracy (solo):** 76–82% on J-UNIWARD stego at 0.4 bpp

---

### 4.4 Branch C — Statistical Features MLP

**Purpose:** Detect classical statistical anomalies (chi-square, histogram)
**Note:** Effective against LSB-specific patterns; weaker against modern adaptive stego

```python
def extract_stat_features(image_array):
    """Extract 64-dim handcrafted statistical feature vector"""
    flat = image_array.flatten()
    features = []
    
    # 1. Pixel value histogram (normalized, 256 bins → summarise to 16)
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    hist_norm = hist / hist.sum()
    features.extend(hist_norm[::16])          # 16 values
    
    # 2. Chi-square statistic per LSB bit plane
    lsb_plane = flat % 2
    expected = np.full(2, len(flat) / 2)
    observed = np.bincount(lsb_plane, minlength=2)
    chi2 = np.sum((observed - expected)**2 / expected)
    features.append(chi2 / len(flat))         # 1 value (normalized)
    
    # 3. Statistical moments
    features.append(float(np.mean(flat)) / 255)
    features.append(float(np.std(flat)) / 255)
    features.append(float(scipy.stats.skew(flat)))
    features.append(float(scipy.stats.kurtosis(flat)))  # 4 values
    
    # 4. Bit plane statistics for 4 LSB planes
    for bit in range(4):
        plane = (flat >> bit) & 1
        features.append(float(np.mean(plane)))
        features.append(float(np.std(plane)))
        features.append(float(scipy.stats.entropy(
            np.bincount(plane, minlength=2) + 1e-10
        )))                                    # 12 values
    
    # 5. Entropy of full image
    features.append(float(scipy.stats.entropy(hist_norm + 1e-10)))  # 1 value
    
    # Pad to exactly 64 features (add zeros if needed)
    features = features[:64]
    features += [0.0] * (64 - len(features))
    return np.array(features, dtype=np.float32)

class BranchC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
    
    def forward(self, x):  # x = 64-dim stat feature vector
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)                  # → 64-dim feature vector
```

---

### 4.5 Fusion Layer

```python
class MBCSSFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = BranchA()  # → 256
        self.branch_b = BranchB()  # → 256
        self.branch_c = BranchC()  # → 64
        
        # Layer norms for each branch output (CRITICAL — prevents domination)
        self.norm_a = nn.LayerNorm(256)
        self.norm_b = nn.LayerNorm(256)
        self.norm_c = nn.LayerNorm(64)
        
        # Fusion classifier
        fused_dim = 256 + 256 + 64  # = 576
        self.fc1  = nn.Linear(fused_dim, 128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2  = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3  = nn.Linear(64, 2)  # [P(clean), P(stego)]
    
    def forward(self, pixel_map, dct_map, stat_feat):
        fa = self.norm_a(self.branch_a(pixel_map))
        fb = self.norm_b(self.branch_b(dct_map))
        fc = self.norm_c(self.branch_c(stat_feat))
        
        fused = torch.cat([fa, fb, fc], dim=1)   # → 576
        x = self.drop1(F.relu(self.fc1(fused)))
        x = self.drop2(F.relu(self.fc2(x)))
        return F.softmax(self.fc3(x), dim=1)      # → [P(clean), P(stego)]
```

**CRITICAL:** LayerNorm before concat. Without this, Branch A (256-dim)
will numerically dominate Branch C (64-dim) and fusion training will fail.

---

### 4.6 Defense Context Scoring

```python
def compute_risk_score(stego_prob, metadata):
    """
    stego_prob: float 0-1 from model output
    metadata: dict with keys: user_privilege, hour, file_size_mb,
              expected_size_mb, destination_type
    """
    risk = stego_prob * 50  # base score from model (max 50)
    
    # Privilege score (0-15)
    if metadata['user_privilege'] == 'intern':     risk += 15
    elif metadata['user_privilege'] == 'employee': risk += 7
    elif metadata['user_privilege'] == 'manager':  risk += 2
    
    # Time anomaly (0-15)
    hour = metadata['hour']
    if hour < 6 or hour > 22:                      risk += 15  # off-hours
    elif hour < 8 or hour > 20:                    risk += 5
    
    # File size anomaly (0-10)
    size_ratio = metadata['file_size_mb'] / (metadata['expected_size_mb'] + 1e-6)
    if size_ratio > 5:    risk += 10
    elif size_ratio > 2:  risk += 5
    
    # Destination (0-10)
    if metadata['destination_type'] == 'external': risk += 10
    elif metadata['destination_type'] == 'dmz':    risk += 5
    
    risk = min(100, risk)
    level = 'CRITICAL' if risk > 75 else 'HIGH' if risk > 50 else 'MEDIUM' if risk > 25 else 'LOW'
    return {'score': round(risk, 1), 'level': level}

# Example output:
# "Image 94.7% stego. User: intern. Time: 02:18 AM. Dest: external.
#  Risk Score: 87.3 / 100 — CRITICAL"
```

---

### 4.7 INT8 Quantization (CPU Deployment)

```python
import torch.quantization

def quantize_model(model, calibration_loader):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)
    
    # Calibration pass (feed 100-200 samples)
    with torch.no_grad():
        for batch in calibration_loader:
            model_prepared(*batch)
    
    model_quantized = torch.quantization.convert(model_prepared)
    return model_quantized

# Expected results:
# Model size: ~50MB → ~12MB (4× reduction)
# CPU inference: ~200ms → ~60ms per image
# Accuracy drop: expect 1–5% (report ACTUAL number; don't assume <2%)
```

---

<a name="prd"></a>
## 5. PRD — PRODUCT REQUIREMENTS DOCUMENT

### 5.1 Functional Requirements

| ID | Requirement | Priority | Acceptance Criterion |
|----|-------------|----------|---------------------|
| FR-01 | Detect LSB steganography in grayscale/color images | MUST | ≥ 82% accuracy on BOSSBase 0.4 bpp |
| FR-02 | Detect DCT/JPEG-domain steganography | MUST | ≥ 76% accuracy on J-UNIWARD 0.4 bpp |
| FR-03 | Detect statistical anomalies (chi-square, histogram) | SHOULD | ≥ 70% on LSB chi-square detection |
| FR-04 | Multi-branch fusion output with confidence score | MUST | 85–90% on mixed stego dataset |
| FR-05 | Defense-context risk scoring (4 factors) | SHOULD | Risk score 0–100, 4 levels |
| FR-06 | INT8 quantized model (CPU-only inference) | MUST | <5% accuracy loss vs full-precision |
| FR-07 | Streamlit web demo (upload → classify) | SHOULD | Functional demo for viva |
| FR-08 | Comparison against Xu-Net baseline | MUST | Table with accuracy, precision, recall, F1 |

### 5.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Inference time (CPU, quantized) | < 500ms per image |
| NFR-02 | Model size (quantized) | < 20 MB |
| NFR-03 | Training environment | Google Colab free T4 GPU |
| NFR-04 | Deployment platform | Any CPU laptop (no GPU required) |
| NFR-05 | Input formats | JPG, PNG, PGM (grayscale or RGB) |
| NFR-06 | Code quality | Well-commented, on GitHub, modular |
| NFR-07 | Report length | 8–12 pages, IEEE double-column format |
| NFR-08 | Plagiarism | < 15% on Turnitin before submission |

### 5.3 Out of Scope (for 3-month student project)
- Real-time video steganalysis
- GAN-generated steganography detection (Branch C is NOT a reliable GAN detector — do NOT claim this)
- Audio steganography
- Network packet steganography
- Adversarial attack resistance
- Multi-GPU distributed training

---

<a name="rules"></a>
## 6. DESIGN DECISIONS & RULES

### 6.1 Architecture Rules (Non-Negotiable)

```
RULE-ARCH-01: Always normalize branch outputs before fusion concat
              Use LayerNorm, NOT BatchNorm (batch size < 8 causes issues)
              Without this: Branch A will dominate → model converges to Branch A only

RULE-ARCH-02: SRM filters in Branch A must be FROZEN during training
              They are precomputed physics-derived kernels
              Allowing gradient flow through them causes instability

RULE-ARCH-03: Train branches SEPARATELY first, THEN fine-tune jointly
              Sequence: (1) Train A alone, (2) Train B alone,
              (3) Train C alone, (4) Freeze branches, train fusion,
              (5) Fine-tune everything together (low LR: 1e-5)

RULE-ARCH-04: Use paired batches during training
              Each mini-batch MUST contain equal cover/stego pairs
              Unpaired batches cause class imbalance during SGD step

RULE-ARCH-05: Branch C is for classical statistical stego, NOT GANs
              Do not claim "GAN detection" in paper/viva — reviewers WILL reject
```

### 6.2 Training Rules

```
RULE-TRAIN-01: Start with 2,000 images on laptop, scale to 10,000 on Colab
RULE-TRAIN-02: Learning rate: 1e-3 for branches, 1e-4 for fusion, 1e-5 for fine-tune
RULE-TRAIN-03: Optimizer: Adam (branches) → SGD with momentum for fine-tuning
RULE-TRAIN-04: Batch size: 32–64 pairs (16–32 cover + 16–32 stego)
RULE-TRAIN-05: Use early stopping (patience=10 epochs on validation set)
RULE-TRAIN-06: Save checkpoint every 5 epochs — Colab disconnects without warning
RULE-TRAIN-07: If loss stops decreasing: check label leakage first (print 10 samples)
               Label leakage = stego images accidentally labeled as clean
RULE-TRAIN-08: Use data augmentation: horizontal flip only (NOT rotation/crop —
               these destroy stego signal and corrupt labels)
```

### 6.3 Evaluation Rules

```
RULE-EVAL-01: ALWAYS compute: Accuracy, Precision, Recall, F1, ROC-AUC
RULE-EVAL-02: Compute PE = ½(P_MD + P_FA) — required for paper comparison
RULE-EVAL-03: Run ablation study: remove Branch B, measure drop; remove Branch C, measure drop
              This proves each branch contributes (required for paper acceptance)
RULE-EVAL-04: Compare against Xu-Net (build simplified version yourself if needed)
RULE-EVAL-05: Test on at least 2 stego algorithms: LSB + S-UNIWARD minimum
RULE-EVAL-06: Report quantization results: size before/after + inference time before/after
RULE-EVAL-07: 99% accuracy claim without ablation = INSTANT reviewer rejection
```

### 6.4 Code Quality Rules

```
RULE-CODE-01: Student A and Student B each have their own Git branch
               Merge to main ONLY after code review
RULE-CODE-02: Every function must have a docstring (professor asks during viva)
RULE-CODE-03: All hyperparameters in config.py (never hardcoded in training script)
RULE-CODE-04: Unit test for LSB embedder: embed → extract → verify identical bits
RULE-CODE-05: Log all metrics to CSV (epoch, train_loss, val_acc, val_pe)
               Colab crashes; logs must survive in Google Drive
```

### 6.5 Patent / Publication Rules

```
RULE-PATENT-01: File provisional patent BEFORE any public disclosure
RULE-PATENT-02: "Any" public disclosure = GitHub repo, college presentation,
                 uploaded report, LinkedIn post, seminar, viva (if public)
RULE-PATENT-03: Check college IP policy BEFORE filing (institution may co-own)
RULE-PATENT-04: Frame claims around COMBINATION, not individual components
RULE-PATENT-05: Patent claims: (a) 3-branch fusion + (b) INT8 CPU deploy + (c) defense-context scoring
RULE-PATENT-06: CRI Guidelines 2025 (effective July 29, 2025) — reference in claim preamble
RULE-PATENT-07: Budget INR 1,750 for provisional; full grant takes 2-4 years
```

---

<a name="checklist"></a>
## 7. IMPLEMENTATION CHECKLIST (Week-by-Week)

### PRE-MONTH 0 (Do This Week)
- [ ] Read Xu-Net paper (Xu et al., 2016, IEEE Signal Processing Letters)
- [ ] Read Yedroudj-Net paper (find on Google Scholar)
- [ ] Create GitHub repo; add both students as collaborators
- [ ] Set up Google Colab + Google Drive sync
- [ ] Talk to professor about project scope approval
- [ ] Talk to college IP cell about patent process (if interested)
- [ ] File provisional patent (if architecture is designed — do NOT wait for results)

### MONTH 1 — FOUNDATIONS

**Week 1 (Student A):**
- [ ] Python: loops, list comprehensions, file I/O, classes
- [ ] NumPy: array ops, slicing, reshape, broadcasting
- [ ] PIL/Pillow: open image, access pixels, convert grayscale, save
- [ ] Git: commit, push, pull, create branch, merge

**Week 1 (Student B):**
- [ ] PyTorch: tensors, autograd, nn.Module, nn.Sequential
- [ ] Training loop template: DataLoader → forward → loss → backward → optimizer.step()
- [ ] Understand what Xu-Net does at a high level (read paper abstract + architecture figure)

**Week 2 (Both):**
- [ ] Implement LSB embedder in Python (test: embed 1000 bits, extract, verify)
- [ ] Implement unit test for LSB: embed → extract → compare (must be 100% match)
- [ ] Compute pixel residual map: `residual = stego_pixels - cover_pixels`
- [ ] Visualize residual: stego vs clean (plot side by side — this becomes Figure 1 in paper)

**Week 3 (Student A):**
- [ ] Download BOSSBase v1.01 from dde.binghamton.edu
- [ ] Embed data into 5,000 images using LSB embedder → stego set
- [ ] Keep 5,000 original → clean set
- [ ] Verify labels: manually inspect 10 stego + 10 clean images, print pixel values
- [ ] Create train/val/test split (70/10/20) → save as CSV (image_path, label)

**Week 4 (Student B):**
- [ ] Build simple 3-layer CNN in PyTorch (baseline before Branch A)
- [ ] Load dataset using custom `torch.utils.data.Dataset` class
- [ ] First training run: 10 epochs, batch=32, Adam lr=1e-3
- [ ] Target: >60% accuracy (proves data pipeline works)
- [ ] Log metrics to CSV, save model checkpoint

**End of Month 1 gate:**
- [ ] Dataset ready (10K images, labels verified)
- [ ] Basic CNN trains and reaches >70% accuracy
- [ ] Both students can explain the full data pipeline (viva-ready)
- [ ] GitHub repo has at least 4 meaningful commits from each student

### MONTH 2 — THREE BRANCHES

**Week 5 (Student A — Branch A):**
- [ ] Load SRM filter weights (from dde.binghamton.edu/download/ — srm_filters.mat)
- [ ] Add SRM layer before CNN (frozen weights, requires_grad=False)
- [ ] Verify residual amplification: plot before/after SRM
- [ ] Train Branch A on LSB dataset
- [ ] Target: ≥82% accuracy (if <75%, check SRM loading)

**Week 5 (Student B — Branch B):**
- [ ] Implement DCT computation on 8×8 image blocks using scipy.fft.dct()
- [ ] Verify DCT output: compare against JPEG decompressed image
- [ ] Build Branch B CNN architecture
- [ ] Train Branch B on J-UNIWARD stego if available (or S-UNIWARD)
- [ ] Target: ≥76% accuracy

**Week 6 (Student A):**
- [ ] Tune Branch A: add/remove conv layers, adjust learning rate
- [ ] Confirm Branch A solo accuracy in writing (for comparison table)
- [ ] Write `branch_a.py` with full docstrings

**Week 6 (Student B):**
- [ ] Tune Branch B, document architecture choices
- [ ] Experiment: 8×8 blocks vs 16×16 blocks vs full-image DCT
- [ ] Write `branch_b.py` with full docstrings

**Week 7 (Both — Branch C):**
- [ ] Implement `extract_stat_features()` (histogram + chi-square + moments)
- [ ] Verify features: stego vs clean feature vectors should differ on chi-square
- [ ] Build Branch C MLP (3 layers: 64→128→64→64)
- [ ] Train Branch C on LSB dataset
- [ ] Target: ≥68% accuracy (this branch is weakest — acceptable)

**Week 8 (Both):**
- [ ] Test all three branches INDEPENDENTLY on test set
- [ ] Write down solo accuracy for each branch (required for ablation study)
- [ ] Merge all three branch files into shared `models/` directory
- [ ] Commit final branch implementations to GitHub

**End of Month 2 gate:**
- [ ] Branch A: ≥82% accuracy (LSB)
- [ ] Branch B: ≥76% accuracy (DCT/JPEG)
- [ ] Branch C: ≥68% accuracy (statistical)
- [ ] Each branch saved as separate .pt file

### MONTH 3 — FUSION + EVALUATION + PAPER

**Week 9 (Both — Fusion):**
- [ ] Implement `MBCSSFusion` class (see architecture section 4.5 above)
- [ ] CRITICAL: Add LayerNorm before concat (RULE-ARCH-01)
- [ ] Load pretrained branch weights into fusion model
- [ ] First fusion run: freeze branches, only train fusion FC layers
- [ ] Target: ≥80% (if lower, check normalization)

**Week 10 (Both — Fine-tuning):**
- [ ] Unfreeze all branches, reduce LR to 1e-5
- [ ] Joint fine-tuning: 20–30 epochs
- [ ] Target: 85–90% on mixed stego (LSB + S-UNIWARD)
- [ ] If 85% → still publish! 85% with 3-branch architecture is strong contribution
- [ ] Implement inference function: takes image path → returns (class, confidence %)

**Week 11 (Student A — Quantization + Evaluation):**
- [ ] Apply INT8 quantization using `torch.quantization`
- [ ] Measure: model size before and after (MB)
- [ ] Measure: inference time before (GPU/CPU) and after (CPU-only)
- [ ] Measure: accuracy before and after quantization
- [ ] Record ALL numbers for paper Table

**Week 11 (Student B — Evaluation Metrics):**
- [ ] Confusion matrix: TP, FP, TN, FN on test set
- [ ] ROC curve: plot and compute AUC score
- [ ] Comparison table vs Xu-Net vs Yedroudj-Net (simplified implementations)
- [ ] Ablation study: run with Branch B removed, Branch C removed → measure impact

**Week 12 (Both — Demo + Report):**
- [ ] Build Streamlit demo (~15 lines):
  ```python
  import streamlit as st
  from PIL import Image
  # load model, predict on uploaded image, show result + confidence
  ```
- [ ] Complete paper draft (8 pages IEEE format)
- [ ] Run Turnitin / plagiarism check — must be < 15%
- [ ] Run Grammarly on entire paper
- [ ] Submit to chosen journal (IETE Journal of Research or DSJ first choice)
- [ ] Final GitHub README with setup instructions

**End of Month 3 gate:**
- [ ] Fusion model achieves ≥85% on mixed stego test set
- [ ] Quantized model: ≤5% accuracy drop, measurably smaller/faster
- [ ] Comparison table complete (your model vs Xu-Net)
- [ ] Ablation table complete (full model vs 2-branch variants)
- [ ] Streamlit demo functional
- [ ] Paper drafted and submitted (or ready to submit)

---

<a name="patent"></a>
## 8. PATENT + PUBLICATION STRATEGY

### 8.1 Patent Eligibility (India, CRI Guidelines 2025)
Your project is patentable because it delivers a **technical effect**:
- Measurably higher detection accuracy (technical improvement)
- 4× smaller model via INT8 (more economical hardware use)
- CPU-only deployment (concrete hardware system improvement)
- Defense-context risk scoring (not a bare algorithm)

**Four Patent Claims (file these exactly as written below):**

**Claim 1 (Broadest):** A computer-implemented system comprising: (a) pixel-residual
CNN branch; (b) DCT-coefficient CNN branch; (c) statistical-feature MLP branch;
(d) fusion layer producing confidence-scored classification output.

**Claim 2 (Defense Scoring):** The system of Claim 1, further comprising a defense-context
scoring module integrating: user privilege, time of transfer, file size anomaly ratio,
and destination network address into a composite risk score.

**Claim 3 (CPU Deployment):** A method of deploying the system of Claim 1 on
CPU-only hardware via INT8 quantization, retaining accuracy within 5% of full-precision model.

**Claim 4 (Method):** A computer-implemented method for detecting insider-threat
data exfiltration via steganographic image files in air-gapped networks.

### 8.2 Filing Timeline
```
IMMEDIATELY:  File provisional patent (Form 2, INR 1,750)
              Website: ipindia.gov.in
              Required: title + description + block diagram of architecture

WITHIN 12 MONTHS: File complete specification (with claims, abstract, drawings)
                   Consult college IP cell — may be free

DO NOT DO BEFORE FILING:
  ✗ Upload project to GitHub (public)
  ✗ Submit paper to journal
  ✗ Present at seminar (open to public)
  ✗ Post on LinkedIn/blog
  ✗ Share in college report portal
```

### 8.3 Journal Target List (Priority Order)

| Priority | Journal | Indexing | Why |
|----------|---------|---------|-----|
| 1st | Defence Science Journal (DSJ) | SCOPUS, WoS | DRDO's own journal, perfect fit, FREE |
| 2nd | IETE Journal of Research | SCOPUS, WoS | India's premier CS/Electronics body |
| 3rd | IET Image Processing | SCOPUS Q2 | Diamond open access (APC = 0) |
| 4th | Journal of Info Security & Applications (Elsevier) | SCOPUS Q1 | For cybersecurity angle |
| 5th | SN Computer Science (Springer) | SCOPUS | Welcomes student authors |
| Dream | IEEE TIFS | SCOPUS Q1 | Hardest but most prestigious |

**Paper must include:**
- Architecture diagram (3-branch CNN figure)
- Ablation study (Table)
- Comparison table vs Xu-Net vs Yedroudj-Net
- ROC curve with AUC value
- Confusion matrix
- Quantization results (size + inference time)
- Dataset description (BOSS, embedding protocol)

---

<a name="risks"></a>
## 9. RISK REGISTER

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| Training loss not decreasing | Medium | High | Check label leakage first; print 10 samples and inspect manually |
| Branch fusion doesn't improve over single branch | Medium | High | Verify LayerNorm before concat; check branch output magnitude imbalance |
| Accuracy below 80% after fusion | Low | Medium | Still publishable if ablation shows fusion > any single branch |
| Colab disconnects during long training | High | Low | Save checkpoint every 5 epochs to Drive; use Colab Pro if needed |
| Dataset too large for laptop | High | Low | Start with 2,000 images (Month 1); use Colab for Month 2 |
| Both students doing same work | Medium | Medium | GitHub branches per student; clear module ownership in README |
| Report pile-up at end | High | High | Write Introduction + Literature Review in Month 1 during training runs |
| Patent invalidated by prior art | Medium | Medium | Claims are on COMBINATION — file narrow, specific claims |
| Paper rejected by journal | Medium | Medium | Submit to DSJ + IETE simultaneously; address review comments |
| INT8 accuracy drop > 5% | Low | Medium | Try FP16 (half-precision) as middle ground; report actual numbers |
| SRM filter loading fails | Low | High | Fallback: initialize Branch A with random weights, train from scratch |
| Professor questions Branch C GAN claim | High | High | Pre-answer: "Branch C targets classical statistical anomalies; Branches A and B handle modern methods" |

---

## APPENDIX A — PAPER READING PRIORITY LIST

1. `Xu et al., 2016` — Xu-Net (baseline you will beat)
2. `Yedroudj et al., 2018` — Yedroudj-Net (second baseline)
3. `Fridrich & Kodovský, 2012` — SRM (how Branch A filters work)
4. `Boroumand et al., 2019` — SRNet (residual design inspiration)
5. `ACM Computing Surveys 2024/25` — Use to justify "generalization gap"
6. `Frontiers AI 2025` — Use to justify "Yedroudj-Net fails after resize"
7. `Uspenskyi & Bondarchuk 2025` — Cite for "resource-constrained steganalysis"
8. `PENet+ 2025 (arXiv:2606.10939)` — Recent lightweight baseline to acknowledge

## APPENDIX B — USEFUL COMMANDS

```bash
# Download BOSSBase (if direct link works)
wget http://dde.binghamton.edu/download/feature_extractors/BOSSbase_1.01.zip

# Check GPU on Colab
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Profile model size
param_count = sum(p.numel() for p in model.parameters())
print(f"Parameters: {param_count:,}")

# Save checkpoint (do this every 5 epochs!)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': val_acc,
}, f'/content/drive/MyDrive/checkpoint_epoch_{epoch}.pt')

# Measure inference time
import time
start = time.time()
with torch.no_grad():
    output = model(pixel_map, dct_map, stat_feat)
elapsed = (time.time() - start) * 1000
print(f"Inference: {elapsed:.1f} ms")
```

## APPENDIX C — 30-SECOND PROFESSOR PITCH

> "Sir, existing steganalysis models (Xu-Net, Yedroudj-Net) work on either
> the pixel domain or the frequency domain — not both simultaneously. We
> propose a parallel three-branch CNN architecture that processes pixel residuals,
> DCT coefficients, and statistical features in parallel, fused via a learned
> fusion layer to produce a confidence-scored classification. We add a
> defense-context scoring module that factors in user privilege, time of
> transfer, and network destination. Finally, INT8 quantization enables
> CPU-only deployment on air-gapped DRDO endpoints with no GPU, retaining
> accuracy within 5% of the full model."

---

*Document Version: 1.0 | Created: September 2026*
*Sources: Project files (July 2026) + Deep research on ACM, IEEE, arXiv, GitHub*
*Legal notes: General information only — consult registered Indian patent agent before filing*
