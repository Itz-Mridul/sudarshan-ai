# Sudarshana — Project Progress Tracker

*Last Updated: 2 October 2026 (Session 2)*

---

## Current State of the Project

### ✅ COMPLETED (28 items)
| # | Item | File(s) | Status |
|---|------|---------|--------|
| 1 | Branch A — Pixel CNN (SRM filters) | `src/branches/branch_a_pixel.py` | ✅ Done |
| 2 | Branch B — DCT Frequency CNN | `src/branches/branch_b_dct.py` | ✅ Done |
| 3 | Branch C — Statistics MLP | `src/branches/branch_c_stats.py` | ✅ Done |
| 4 | **Branch D — Generative AI Detector** | `src/branches/branch_d_generative.py` | ✅ Done |
| 5 | Fusion Model (3-Branch + Dense/CrossAttn) | `src/model/fusion_model.py` | ✅ Done |
| 6 | **Cross-Attention Transformer Fusion** | `src/model/cross_attention_fusion.py` | ✅ Done |
| 7 | **Open-Set Detector (Energy + Softmax)** | `src/model/open_set_detector.py` | ✅ Done |
| 8 | SRM Filter Layer (30 kernels) | `src/model/srm_filters.py` | ✅ Done |
| 9 | LSB Embedder (dataset generation) | `src/data_pipeline/lsb_embedder.py` | ✅ Done |
| 10 | DCT Embedder (dataset generation) | `src/data_pipeline/dct_embedder.py` | ✅ Done |
| 11 | Dataset Loader + Augmentation | `src/data_pipeline/dataset.py` | ✅ Done |
| 12 | Training Loop (AdamW, AMP, Cosine LR) | `src/training/train.py` | ✅ Done |
| 13 | Training Orchestrator (3-stage) | `train_all.py` | ✅ Done |
| 14 | INT8 Quantization for Edge Deploy | `src/deploy/quantize.py` | ✅ Done |
| 15 | Streamlit Demo App (6 tabs, full UI) | `app/demo.py` | ✅ Done |
| 16 | **Grad-CAM Explainability (Tab 6)** | `src/utils/gradcam.py` + `app/demo.py` | ✅ Done |
| 17 | **Explanation Stability Testing** | `app/demo.py` (tab 6) | ✅ Done |
| 18 | Baseline: Xu-Net | `src/baselines/xunet_baseline.py` | ✅ Done |
| 19 | Baseline: SRNet | `src/baselines/srnet_baseline.py` | ✅ Done |
| 20 | **8-Way Ablation Study Script** | `ablation_study.py` | ✅ Done |
| 21 | Benchmark Inference Script | `benchmark_inference.py` | ✅ Done |
| 22 | Risk Scoring Engine | `src/deploy/risk_scoring.py` | ✅ Done |
| 23 | **FastAPI REST API** | `src/deploy/api.py` | ✅ Done |
| 24 | **Dockerfile (+ Jetson support)** | `Dockerfile` | ✅ Done |
| 25 | Pre-trained Weights (untrained init) | `weights/fusion_best.pt`, `fusion_int8.pt` | ✅ Done |
| 26 | Revised IEEE Implementation Plan | `StegShield_Implementation_Plan_v2.md` | ✅ Done |
| 27 | DRDO/ISRO Action Plan | `DRDO_ISRO_Action_Plan.md` | ✅ Done |
| 28 | Progress Tracker | `PROGRESS_TRACKER.md` | ✅ Done |

| 29 | **ALASKA2 Dataset Integration** | `scripts/setup_alaska2.py` | ✅ Done |
| 30 | **Literature Gap Comparison Table** | `LITERATURE_GAP_ANALYSIS.md` | ✅ Done |
| 31 | **IEEE Paper Draft (LaTeX)** | `IEEE_Paper_Draft.tex` | ✅ Done |

### 📋 REMAINING TODO
| # | Item | Priority | Est. Time |
|---|------|----------|-----------|
| 32 | Train on real dataset (Colab/RTX 5060) | HIGH | Ongoing |

---

## Architecture (Complete)

```
              Input Image
              │
    ┌─────────┼─────────┬──────────┐
    ▼         ▼         ▼          ▼
 Branch A  Branch B  Branch C  Branch D
 Pixel/SRM DCT/Freq  Stats    Generative
 256-dim   256-dim   32-dim   128-dim
    │         │         │          │
    ▼         ▼         ▼          ▼
 ┌──────────────────────────────────────┐
 │  Cross-Attention Transformer Fusion  │
 │  (or Dense Fusion — selectable)      │
 └───────────────┬──────────────────────┘
                 ▼
          Classification
         [CLEAN / STEGO / UNKNOWN]
              │         │
       Grad-CAM     Open-Set
       Heatmap      Detector
```
