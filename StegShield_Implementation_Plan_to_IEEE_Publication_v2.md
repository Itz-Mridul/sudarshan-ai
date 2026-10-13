# StegShield — Implementation Plan to IEEE Publication (Revised)

*This document outlines the revised, five-phase plan to bring the StegShield project to an IEEE-level publication. It prioritizes research-gap verification, experimental controls, open-set evaluation, and rigorous evidence requirements.*

---

## 1. Research Question and Intended Contributions

Before planning the experiments, the project is guided by a formal research question:

> **"Can StegShield detect conventional and generative image steganography when the test-time embedding algorithm or generative model was not seen during training, and can it identify unfamiliar cases under changes in image source or processing conditions?"**

### Intended Contributions
Until experiments confirm the outcomes, the following are the *proposed* contributions of the paper:
1. **The Detection Framework:** A multi-branch architecture attempting to unify spatial, frequency, and statistical steganalysis.
2. **The Held-Out-Method Protocol:** Evaluation against unseen conventional and generative algorithms to test generalizability.
3. **Domain-Shift Experiments:** Independent testing across changes in image source, compression, format, and color characteristics.
4. **Uncertainty-Handling Mechanism:** An open-set recognition approach to identify unfamiliar steganographic methods rather than forcing a known classification.
5. **Evidence-Based Explanation System:** Explainability maps (e.g., Grad-CAM) tested for decision stability and correspondence.

### The Novelty Claim
The novelty of this project will not be assumed beforehand. **This research investigates whether a unified image-steganalysis framework can detect conventional and generative steganography under unseen algorithms or generative models, while identifying unfamiliar cases and evaluating generalization across image sources and processing conditions.** The final novelty claim will be established through the focused literature review (Section 3) and will describe what StegShield demonstrably does better, not what the team hopes it will achieve.

---

## 2. Phase 1 — Finish Training the Existing 3-Branch Model

*Do not add more architecture merely to make the project appear novel. Establish reliable results for the existing three-branch model first.*

1. **Dataset Pipeline:** Download BOSSbase 1.01 (10,000 grayscale PGM images, 512x512). It is the standard dataset for Xu-Net, Yedroudj-Net, and SRNet.
2. **Data Splits & Leakage Prevention:** Run the existing pipeline end-to-end. Ensure that clean and stego versions of the *same* image land in the *same* split (train/val/test) to prevent data leakage. Document image dimensions, payload settings, preprocessing, and splits.
3. **Train the Core:** Train Branch A, B, and C individually, followed by the fused model. Use WandB for logging.
4. **Measure, Don't Assume:** The target numbers in the spec (Branch A ≥82%, fused model ≥85-90%, etc.) must remain clearly labelled as *goals* until measured. They must never appear as experimental findings before training.
5. **Baselines:** Train the Xu-Net baseline on the exact same data split. If they can be reproduced fairly, include **SRNet** or **Yedroudj-Net** as well.
6. **Payload-Sensitivity Testing:** Test at 0.4, 0.2, 0.1, and 0.05 bits-per-pixel.
7. **Record Variability:** Include multiple runs or different random seeds when feasible, and record variability in the results.

---

## 3. Dedicated Research-Gap Verification

Before designing a fourth branch for generative steganography, the team must determine whether the proposed gap is genuinely underexplored.

**Action:** Conduct a focused literature search across IEEE Xplore, ACM Digital Library, SpringerLink, ScienceDirect, and arXiv. 

Create a formal comparison table cross-referencing recent papers against the following capabilities:
* Conventional Steganography Detection
* GAN-based Steganography Detection
* Diffusion-based Steganography Detection
* Unseen-Algorithm Testing
* Open-Set Recognition
* Cross-Dataset Testing
* Uncertainty Estimation
* Explainability

**This comparison table is an explicit deliverable.** Only commit to building Branch D (Phase 4) if this table proves the gap is real and the combination of features is genuinely missing from the literature.

---

## 4. Phase 2 — The 4th Branch Hypothesis (GAN + Diffusion)

**Do not assume that adding a GAN/diffusion branch will improve StegShield. Treat this branch as an experimental hypothesis.**

1. **Clean-Generated Controls:** For coverless/diffusion methods that generate entirely new images, you **must** include suitable clean-generated control images from the *same* generation process (carrying no hidden payload). Otherwise, the model will simply learn to distinguish "AI-generated vs. Camera photograph" instead of detecting hidden information.
2. **Experimental Comparison:** Compare the existing 3-branch model against the expanded 4-branch model using the exact same training and test conditions.
3. **The Bar for Inclusion:** Measure whether the fourth branch improves generative-steganography detection *without* damaging conventional-steganography detection. Keep the fourth branch only if the experimental results justify its additional complexity.

---

## 5. Phase 3 — Evaluation: Unseen-Algorithm, Domain-Shift, and Open-Set

### A. Independent Domain-Shift Testing
Do not test multiple domain shifts at once, as performance drops will have ambiguous causes. Test these factors separately wherever possible:
1. **Image Source:** Compare different image sources (e.g., BOSSbase vs. BOWS2) under comparable conditions.
2. **Transformations:** Evaluate format (JPEG vs PNG), color, compression, and other transformations (resizing, cropping) independently.

### B. Rigorous Open-Set Evaluation
The model should output "unknown/uncertain" for unfamiliar methods, but this requires a precise methodology:
1. **Thresholding:** Precisely explain how the model decides a sample is unknown. The threshold must be selected using **validation data**, not the final test set.
2. **Three Separate Outcomes:** Evaluate and report:
   * How often unseen methods are correctly flagged as unknown.
   * How often known methods are incorrectly flagged as unknown.
   * How accurately the model classifies known methods.
*Note: Without this exact experiment, the project must describe itself as a multi-class detector, not an open-set detector.*

### C. Comprehensive Metrics
Report Accuracy, Precision, Recall, F1-score, and ROC-AUC. Include Equal Error Rate (EER) where appropriate.

---

## 6. Phase 4 — Rigorous Explainability and Ablation

A heatmap that looks convincing is not sufficient evidence that the model is using meaningful steganographic traces.

1. **Explanation Stability & Correspondence:** Test whether the explanations are stable. Check whether the explanation changes appropriately when the relevant input evidence is altered or removed. The explanation must reflect the model's actual decision pathway.
2. **Full Ablation:** Execute the 8-way ablation study. Prove that removing each branch or fusion mechanism hurts performance, and provide a one-line reason *why* for each row in the table.

---

## 7. Phase 5 — Venue Planning and Writing

Do not write the Introduction and Abstract until the experiments are complete and the actual contribution is known. Overpromising in the abstract guarantees reviewer rejection.

### Potential IEEE Venues
Keep the following as possible venues, rather than guaranteeing any one is suitable. Distinguish confirmed submission deadlines from estimates, and check official conference calls before committing. Final selection depends on the completed research scope and results.

* **IEEE WIFS 2027:** (Information Forensics and Security). Excellent topical fit.
* **IEEE ICIP 2027:** (Image Processing). Broad scope, flagship conference.
* **IEEE Access:** Rapid, open-access journal option if the paper size exceeds conference limits.

---

## 8. Final Submission Checklist (Evidence Requirements)

These are formal conditions for submitting the paper. The team must verify that:

- [ ] Every reported number comes from an actual experiment; no target values or goals are left in the paper as results.
- [ ] The data split completely prevents leakage (clean and stego versions of an image are in the same split).
- [ ] Baselines (Xu-Net, SRNet/Yedroudj-Net) are trained under comparable protocols.
- [ ] The fourth branch demonstrates a measurable benefit over the 3-branch model (if included).
- [ ] For coverless steganography, matched clean-generated controls were used.
- [ ] Unseen methods were genuinely withheld from training.
- [ ] If claiming open-set behavior, the unknown threshold was set via validation data, and false-unknown rates are reported.
- [ ] Explanation outputs (heatmaps) were tested for stability and correspondence to the actual decision.
- [ ] A related-work comparison table is included, built from a focused literature search to prove the research gap.
- [ ] Limitations and failure cases are reported honestly and prominently.
