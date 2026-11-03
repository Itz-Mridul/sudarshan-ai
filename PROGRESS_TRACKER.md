# Sudarshana (StegShield) — Project Progress Tracker

*Last Updated: 4 November 2026*

---

## Current State of the Project

### ✅ COMPLETED (40 items)
| # | Item | File(s) | Status |
|---|------|---------|--------|
| 1 | Branch A — Pixel CNN (30 SRM filters) | `src/branches/branch_a_pixel.py` | ✅ Done |
| 2 | Branch B — DCT Frequency CNN | `src/branches/branch_b_dct.py` | ✅ Done |
| 3 | Branch C — Statistics MLP | `src/branches/branch_c_stats.py` | ✅ Done |
| 4 | Branch D — Generative AI Detector (Depthwise SE) | `src/branches/branch_d_generative.py` | ✅ Done |
| 5 | Fusion Model (4-Branch Dense/CrossAttn) | `src/model/fusion_model.py` | ✅ Done |
| 6 | Cross-Attention Transformer Fusion | `src/model/cross_attention_fusion.py` | ✅ Done |
| 7 | Open-Set Detector (Helmholtz Energy + Softmax) | `src/model/open_set_detector.py` | ✅ Done |
| 8 | SRM Filter Layer (30 high-pass kernels) | `src/model/srm_filters.py` | ✅ Done |
| 9 | LSB Embedder (dataset generation) | `src/data_pipeline/lsb_embedder.py` | ✅ Done |
| 10 | DCT Embedder (capacity-safe JPEG generation) | `src/data_pipeline/dct_embedder.py` | ✅ Done |
| 11 | Dataset Loader + Pair-Safe Augmentation | `src/data_pipeline/dataset.py` | ✅ Done |
| 12 | Training Loop (AdamW, AMP, Cosine LR) | `src/training/train.py` | ✅ Done |
| 13 | Training Orchestrator (3-stage) | `train_all.py` | ✅ Done |
| 14 | INT8 Dynamic Quantization (QNNPACK/ARM) | `src/deploy/quantize.py` | ✅ Done |
| 15 | Streamlit Forensic Workbench App (6 tabs) | `app/demo.py` | ✅ Done |
| 16 | Grad-CAM Explainability Engine | `src/utils/gradcam.py` + `app/demo.py` | ✅ Done |
| 17 | Explanation Stability & Faithfulness Metric | `app/demo.py` (tab 6) | ✅ Done |
| 18 | Baseline: Xu-Net | `src/baselines/xunet_baseline.py` | ✅ Done |
| 19 | Baseline: SRNet | `src/baselines/srnet_baseline.py` | ✅ Done |
| 20 | 8-Way Architectural Ablation Study | `ablation_study.py` | ✅ Done |
| 21 | Edge Benchmark Inference Script | `benchmark_inference.py` | ✅ Done |
| 22 | Forensic Risk Scoring Engine | `src/deploy/risk_scoring.py` | ✅ Done |
| 23 | FastAPI Production REST API | `src/deploy/api.py` | ✅ Done |
| 24 | Dockerfile (+ Jetson / ARM support) | `Dockerfile` | ✅ Done |
| 25 | Pre-trained Weights (Colab & Local) | `weights/*.pt` | ✅ Done |
| 26 | Revised IEEE Implementation Plan | `StegShield_Implementation_Plan_v2.md` | ✅ Done |
| 27 | DRDO/ISRO Action Plan | `DRDO_ISRO_Action_Plan.md` | ✅ Done |
| 28 | Progress Tracker | `PROGRESS_TRACKER.md` | ✅ Done |
| 29 | ALASKA2 Dataset Integration | `scripts/setup_alaska2.py` | ✅ Done |
| 30 | Literature Gap Comparison Table | `LITERATURE_GAP_ANALYSIS.md` | ✅ Done |
| 31 | IEEE Transactions Manuscript Draft | `IEEE_Paper_Draft.tex` | ✅ Done |
| 32 | IEEE BibTeX Bibliography Database | `references.bib` | ✅ Done |
| 33 | Rigorous Research Protocol (TIFS/CCS) | `RIGOROUS_RESEARCH_PROTOCOL_TIFS.md` | ✅ Done |
| 34 | Faithfulness & Ground-Truth IoU Evaluator | `experiments/eval_faithfulness.py` | ✅ Done |
| 35 | Energy-Based OOD Evaluation Suite | `experiments/eval_ood.py` | ✅ Done |
| 36 | Inference Footprint Benchmark Suite | `benchmark_inference.py` & `results/benchmark_results.json` | ✅ Done |
| 37 | Architectural Ablation Experiment Suite | `ablation_study.py` & `results/ablation_results.json` | ✅ Done |
| 38 | Publication Figure Generator | `generate_paper_figures.py` & `results/paper_figures/` (Fig 1–4) | ✅ Done |
| 39 | Research Papers & References Specification | `RESEARCH_PAPERS_AND_REFERENCES.txt` | ✅ Done |
| 40 | Unit Test Suite (26/26 Passing) | `run_all_tests.py` | ✅ Done |

---

## Verified Research Findings

### 1. Edge Inference Footprint & Latency (Apple Silicon / ARM)
- **Branch A (Pixel CNN):** 13.04 MB, 3.25M parameters, 146.2 ms median latency.
- **Branch B (DCT CNN):** 5.76 MB, 1.43M parameters, 77.7 ms median latency.
- **Branch C (Stats MLP):** 0.43 MB, 105K parameters, 7.9 ms median latency.
- **MBCSS Fusion FP32:** 21.31 MB, 5.30M parameters, 254.3 ms median latency.
- **MBCSS Fusion INT8:** 1.16 MB, 4.69M parameters, 246.7 ms median latency ($18.4\times$ footprint reduction).

### 2. Attribution Faithfulness (RQ4)
- **Mean Ground-Truth Mask IoU:** $0.0951 \pm 0.0017$ against true bit modification masks.
- **Progressive Deletion Drop:** Top 20% mask drop = 1.31%, Top 50% mask drop = 2.45%.

### 3. Open-Set Out-of-Distribution Detection (RQ3)
- **Benign Noisy Covers ($\sigma=0.05$):** Helmholtz Free Energy achieved **0.92 AUROC** (vs. 0.68 for Softmax) with **FPR95 of 0.20** (vs. 0.40 for Softmax).
- **Energy Superiority Confirmed on Domain Shift / Sensor Noise.**

---

## Architectural Topology

```
                         Input Carrier Image
                                  │
         ┌──────────────┬─────────┴─────────┬──────────────┐
         ▼              ▼                   ▼              ▼
     Branch A       Branch B            Branch C       Branch D
    Pixel / SRM     Block-DCT          Stats & RS      Generative
     (256-dim)      (256-dim)           (32-dim)       (128-dim)
         │              │                   │              │
         └──────────────┼───────────────────┴──────────────┘
                        ▼
      ┌────────────────────────────────────┐
      │  LayerNorm + Multi-Head Attention  │
      │  (Cross-Attention / Dense Fusion)  │
      └─────────────────┬──────────────────┘
                        ▼
                 Class Logits f(x)
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
   Softmax Probabilities       Helmholtz Free Energy
  [CLEAN / STEGO Decision]     E(x; T) vs Threshold tau
         │                             │
         ▼                             ▼
   Grad-CAM Saliency           [ACCEPTED / REJECTED OOD]
   & Faithfulness Audit
```
