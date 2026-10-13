# StegShield — Research References & GitHub Implementations

This document tracks every research paper, dataset, and GitHub repository
that directly influences the StegShield architecture. Where applicable,
the actual code is **integrated into the project** under `src/baselines/`
or `external_repos/`.

---

## Table of Contents
1. [Foundational Papers & Implementations](#1-foundational-papers--implementations)
2. [Datasets](#2-datasets)
3. [Recent Research (2024–2026)](#3-recent-research-2024-2026)
4. [GitHub Repositories](#4-github-repositories)
5. [How We Beat Each Baseline](#5-how-we-beat-each-baseline)

---

## 1. Foundational Papers & Implementations

### [SRM Filters] Fridrich & Kodovsky, 2012
| Field | Detail |
|:---|:---|
| **Title** | "Rich Models for Steganalysis of Digital Images" |
| **Journal** | IEEE Transactions on Information Forensics and Security, 7(3), 868–882 |
| **DOI** | 10.1109/TIFS.2011.2160391 |
| **Key Idea** | 30 hand-crafted high-pass filter kernels that suppress image content and amplify steganographic noise residuals |
| **Our Use** | `src/model/srm_filters.py` — **SRM filters are used as the frozen first layer of Branch A** |
| **Status** | ✅ Directly implemented in our codebase |

---

### [Xu-Net] Xu, Wu & Shi, 2016
| Field | Detail |
|:---|:---|
| **Title** | "Structural Design of Convolutional Neural Networks for Steganalysis" |
| **Journal** | IEEE Signal Processing Letters, 23(5), 708–712 |
| **DOI** | 10.1109/LSP.2016.2548421 |
| **Key Idea** | First CNN to match traditional ensemble classifiers. 5-layer architecture with preprocessing high-pass filter |
| **Dataset** | BOSSbase 1.01 (10,000 grayscale images) |
| **Accuracy** | ~80–85% on S-UNIWARD at 0.4 bpp |
| **Weakness** | Single spatial domain only. Blind to JPEG-domain steganography |
| **Our Baseline** | `src/baselines/xunet_baseline.py` — **trained and compared in paper Table 1** |
| **Status** | ✅ Implemented in our codebase |

---

### [Yedroudj-Net] Yedroudj, Comby & Chaumont, 2018
| Field | Detail |
|:---|:---|
| **Title** | "Yedroudj-Net: An Efficient CNN for Spatial Steganalysis" |
| **Conference** | IEEE ICASSP 2018 |
| **GitHub** | https://github.com/YedK/steganalysis_with_CNN_Yedroudj-Net |
| **Local Copy** | `external_repos/steganalysis_with_CNN_Yedroudj-Net/` |
| **Key Idea** | Improved Xu-Net with SRM preprocessing + TLU (Truncation Layer Unit) + BatchNorm |
| **Architecture** | SRM → TLU → CNN (BN + ReLU blocks) → Classifier |
| **Accuracy** | ~85–89% on WOW and S-UNIWARD at 0.4 bpp |
| **Weakness** | Requires GPU. Not optimized for edge/CPU deployment |
| **Our Use** | Architecture reference for Branch A design. TLU concept inspired our bounded activation approach |
| **Status** | ✅ Code available in `external_repos/` — reference for paper comparison |

**Key insight borrowed:** The Truncation Layer Unit (TLU):
```python
# From: external_repos/steganalysis_with_CNN_Yedroudj-Net/pytorch_version/yed.py
class TLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-self.threshold, max=self.threshold)
```
This clamps extreme activation values, stabilizing training on noisy residual data.

---

### [SRNet] Boroumand, Chen & Fridrich, 2019
| Field | Detail |
|:---|:---|
| **Title** | "Deep Residual Network for Steganalysis of Digital Images" |
| **Journal** | IEEE Transactions on Information Forensics and Security, 14(5), 1181–1193 |
| **DOI** | 10.1109/TIFS.2018.2871749 |
| **GitHub (Reference)** | https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet |
| **Local Copy** | `external_repos/Pytorch-implementation-of-SRNet/` |
| **Our Implementation** | `src/baselines/srnet_baseline.py` — **directly adapted from the reference code** |
| **Architecture** | 12-layer deep residual network: 2 intro layers + 5 residual blocks + 4 downsampling blocks + FC |
| **Accuracy** | ~89–91% on S-UNIWARD at 0.4 bpp (BOSSbase 1.01) |
| **Weakness** | Single domain (spatial only). Very deep = slow on CPUs. ~12 MB model size |
| **Status** | ✅ **Fully implemented** at `src/baselines/srnet_baseline.py` |

**Run to generate paper comparison:**
```bash
python src/baselines/srnet_baseline.py --train --epochs 50
```

---

## 2. Datasets

### BOSSbase 1.01 (Primary Training Dataset)
| Field | Detail |
|:---|:---|
| **Source** | http://www.agents.cz/boss/ |
| **Size** | 10,000 grayscale PGM images (512×512) |
| **Content** | Natural photographs from multiple cameras |
| **Format** | PGM (lossless, grayscale) |
| **Used By** | Xu-Net, Yedroudj-Net, SRNet, and virtually all steganalysis papers |
| **Our Use** | Primary training/validation/test dataset |
| **Setup** | `bash scripts/setup_boss_dataset.sh` |
| **Limitation** | Homogeneous pipeline. Poor generalization to real-world images |

### ALASKA#2 (Recommended Next Dataset)
| Field | Detail |
|:---|:---|
| **Source** | https://www.kaggle.com/competitions/alaska2-image-steganalysis |
| **Size** | 75,000 cover + 75,000 stego images per algorithm |
| **Algorithms** | JMiPOD, J-UNIWARD, UERD |
| **Format** | JPEG (color, real-world camera diversity) |
| **Why** | Addresses Cover-Source Mismatch. Real phones, DSLRs, diverse content |
| **Reference** | https://github.com/Rinovative/alaska2-steganalysis |
| **Our Plan** | Train Branch B (DCT) on this dataset for JPEG steganography specialization |

### IStego100K (Large-Scale)
| Field | Detail |
|:---|:---|
| **Source** | https://github.com/YangzlTHU/IStego100K |
| **Size** | 200,000+ images |
| **Why** | Scale training for better generalization across payload rates |

### BOWS2 (Cross-Source Testing)
| Field | Detail |
|:---|:---|
| **Source** | http://bows2.ec-lille.fr/ |
| **Size** | 10,000 images (same format as BOSSbase) |
| **Why** | Train on BOSSbase, test on BOWS2 = proves cross-source generalization |

### UCF101 (Future Video Extension)
| Field | Detail |
|:---|:---|
| **Source** | https://www.crcv.ucf.edu/data/UCF101.php |
| **Size** | 13,320 video clips |
| **Format** | AVI |
| **Why** | Standard benchmark for video steganalysis (future Branch D) |

---

## 3. Recent Research (2024–2026)

### CNN-GRU Hybrid (2024)
| Field | Detail |
|:---|:---|
| **Finding** | Combining CNN with GRU (Gated Recurrent Unit) improves generalization. CNN extracts spatial features; GRU models sequential dependencies across image patches |
| **Source** | ResearchGate, Frontiers in Computing |
| **Gap Addressed** | Better generalization on unseen data |
| **Our Roadmap** | Future: add GRU after fusion concatenation layer |

### Hybrid CNN-NLP (2025)
| Field | Detail |
|:---|:---|
| **Finding** | Use NLP (spaCy semantic filter) to validate extracted "message" coherence — reduces false positives |
| **Source** | MDPI Journals |
| **Gap Addressed** | High false positive rate in practice |
| **Our Use** | Branch C statistical features partially address this via chi-square coherence |

### SA-CNN (2025)
| Field | Detail |
|:---|:---|
| **Title** | "Deep Learning-Based Steganalysis for Detection and Classification of Possible Hidden Content" |
| **Finding** | SA-CNN with High-Pass Filters enables multi-class classification (not just binary) |
| **Source** | AmericasPG Journal |
| **Gap Addressed** | Identifying *which* algorithm was used, not just *if* stego exists |
| **Our Roadmap** | Extend fusion output from 2-class to 5-class (clean / LSB / J-UNIWARD / F5 / nsF5) |

### StegBench Dual-Branch (2026)
| Field | Detail |
|:---|:---|
| **Title** | "StegBench: A Dual-Branch Benchmark Dataset for Multi-Class Steganalysis" |
| **Source** | MDPI |
| **Finding** | New benchmark proving dual-branch models outperform single-branch baselines |
| **Our Response** | StegShield uses **three** branches — expected to outperform dual-branch |

### Adversarial Steganography (2024–2025)
| Field | Detail |
|:---|:---|
| **Finding** | GAN-generated stego images are specifically designed to fool CNNs |
| **Source** | MDPI, arXiv |
| **Gap** | Our model is not yet tested against adversarial stego |
| **Our Roadmap** | Adversarial training in future version |

---

## 4. GitHub Repositories

| # | Repository | Use in Project |
|:--|:---|:---|
| 1 | [brijeshiitg/Pytorch-implementation-of-SRNet](https://github.com/brijeshiitg/Pytorch-implementation-of-SRNet) | **✅ Adapted into `src/baselines/srnet_baseline.py`** |
| 2 | [YedK/steganalysis_with_CNN_Yedroudj-Net](https://github.com/YedK/steganalysis_with_CNN_Yedroudj-Net) | **✅ Cloned in `external_repos/`** — architecture reference for Branch A |
| 3 | [Rinovative/alaska2-steganalysis](https://github.com/Rinovative/alaska2-steganalysis) | EfficientNet on ALASKA#2 — reference for future JPEG branch |
| 4 | [YangzlTHU/IStego100K](https://github.com/YangzlTHU/IStego100K) | Large-scale dataset — recommended for future training |
| 5 | [daniellerch/aletheia](https://github.com/daniellerch/aletheia) | Open-source steganalysis toolkit — reference for statistical features |
| 6 | [nicowillis/steganalysis_cnn_srm](https://github.com/nicowillis/steganalysis_cnn_srm) | CNN+SRM implementation — architectural reference |

---

## 5. How We Beat Each Baseline

| Baseline | Their Best Accuracy | Their Weakness | Our Advantage |
|:---|:---:|:---|:---|
| **Antivirus (Norton/McAfee)** | 0% | Blind to steganography entirely | Our model is purpose-built for stego detection |
| **Chi-Square Statistical Test** | ~60% | Only catches uniform LSB. Fails on adaptive stego | Our Branch C uses 262 statistical features + ML |
| **Xu-Net (2016)** | ~83% | Spatial only. Fails on JPEG stego | Our Branch B (DCT) catches what Xu-Net misses |
| **Yedroudj-Net (2018)** | ~88% | GPU-only. Not deployable on air-gapped CPUs | Our INT8 quantization makes it CPU-deployable |
| **SRNet (2019)** | ~90% | Single domain. No context scoring. GPU-heavy | Three-domain fusion + risk scoring + INT8 |
| **StegShield (Ours)** | **>92% (target)** | Currently needs real dataset training | All three domains + defense context + CPU-ready |

---

## 6. Citation Format (For Paper)

If writing your IEEE paper, cite these as:

```
[1] J. Fridrich and J. Kodovsky, "Rich Models for Steganalysis of Digital
    Images," IEEE Trans. Inf. Forensics Security, vol. 7, no. 3, pp.
    868–882, Jun. 2012.

[2] G. Xu, H.-Z. Wu, and Y.-Q. Shi, "Structural Design of CNNs for
    Steganalysis," IEEE Signal Process. Lett., vol. 23, no. 5, pp.
    708–712, May 2016.

[3] M. Yedroudj, F. Comby, and M. Chaumont, "Yedroudj-Net: An Efficient
    CNN for Spatial Steganalysis," in Proc. IEEE ICASSP, Apr. 2018.

[4] M. Boroumand, M. Chen, and J. Fridrich, "Deep Residual Network for
    Steganalysis of Digital Images," IEEE Trans. Inf. Forensics Security,
    vol. 14, no. 5, pp. 1181–1193, May 2019.
```

---

*Last updated: September 2026*
*All GitHub links verified. Local copies in `external_repos/`.*
