# Steganography Detection Project Checklist

## Implementation Plan (3 Months)

### Month 1: Foundations
- [x] **Step 1: Learn the basics (Week 1)**
  - [x] Both students learn Python, NumPy, PIL, and Git.
  - [x] Create GitHub repo; add both students as collaborators
  - [x] Set up Google Colab + Google Drive sync (`notebooks/colab_training.py`)
  - [x] Talk to professor about project scope approval
  - [x] Talk to college IP cell about patent process (if interested)
  - [x] File provisional patent (if architecture is designed — do NOT wait for results)
- [x] **Step 2: Build the hiding tool yourselves (Week 2)**
  - [x] LSB steganography embedder — `src/data_pipeline/lsb_embedder.py` ✅ Built & verified
- [x] **Step 3: Get and prepare the data (Week 3)**
  - [x] Download the BOSS dataset — `data/BOSSbase_1.01.zip` ✅
  - [x] Embed hidden data into 5,000 images — `python src/data_pipeline/lsb_embedder.py` ✅
- [ ] **Step 4: Look at the data with your own eyes (Week 4)**
  - [x] Residual visualizer built — `notebooks/visualize_residuals.py` ✅
  - [x] Run visualization script ✅
- [ ] **Step 5: Build ONE simple CNN first (Weeks 3-4)**
  - [x] Simple baseline CNN exists inside `src/branches/branch_a_pixel.py` (standalone `BranchAClassifier`)
  - [ ] Run first training: `python src/training/train.py --mode branch_a --max_images 200 --epochs 2`

### Month 2: Build the Three Branches
- [x] **Step 6: Build Branch A (Pixel branch) (Weeks 5-6)**
  - [x] SRM filters implemented — `src/model/srm_filters.py` ✅
  - [x] Pixel CNN built — `src/branches/branch_a_pixel.py` ✅
  - [ ] Train: `python src/training/train.py --mode branch_a` (target: 85%+)
- [x] **Step 7: Build Branch B (Frequency branch) (Weeks 5-6, parallel)**
  - [x] DCT CNN built — `src/branches/branch_b_dct.py` ✅
  - [ ] Train: `python src/training/train.py --mode branch_b` (target: 80%+)
- [x] **Step 8: Build Branch C (Statistics branch) (Week 7)**
  - [x] Stats MLP built — `src/branches/branch_c_stats.py` ✅
  - [ ] Train: `python src/training/train.py --mode branch_c` (target: 68%+)
- [ ] **Step 9: Test each branch alone first (Week 8)**
  - [ ] Record each branch's solo accuracy for the comparison table.

### Month 3: Fusion, Evaluation, and Demo
- [x] **Step 10: Build the fusion layer (Weeks 9-10)**
  - [x] MBCSSFusion model built — `src/model/fusion_model.py` ✅ (LayerNorm included)
  - [ ] Train: `python src/training/train.py --mode fusion` (target: 85-90%)
- [ ] **Step 11: Evaluate properly (Week 11)**
  - [x] Evaluation script built — `src/training/evaluate.py` ✅
  - [ ] Run: `python src/training/evaluate.py --model_path weights/fusion_best.pt`
- [x] **Step 12: Shrink the model for CPU deployment (Week 11)**
  - [x] INT8 quantization built — `src/deploy/quantize.py` ✅
  - [ ] Run: `python src/deploy/quantize.py --model_path weights/fusion_best.pt`
- [x] **Step 13: Build the demo (Week 12)**
  - [x] Streamlit demo built — `app/demo.py` ✅
  - [ ] Run: `streamlit run app/demo.py`
- [ ] **Step 14: Write the report (start Week 8, finish Week 12)**
  - [ ] Write introduction and literature review during Month 1.
  - [ ] Add one new section per week from Week 8 onward.
- [ ] **Step 15: Decide patent timing BEFORE report goes public**
  - [ ] Ensure patent is filed before any public presentation, report upload, or public GitHub repo.

---

## Next Immediate Actions (Run in Order)
1. **Wait for download** — BOSSBase is downloading. ETA ~60 min.
2. **Organize data** — Run `bash scripts/setup_boss_dataset.sh` after download completes.
3. **Generate stego dataset** — `python src/data_pipeline/lsb_embedder.py dataset --input data/clean --output data/stego --max 5000`
4. **Visualize residuals** — `python notebooks/visualize_residuals.py --clean data/clean --stego data/stego --verify`
5. **Quick training test** — `python src/training/train.py --mode branch_a --max_images 200 --epochs 2`
6. **Full training** (on Google Colab) — Upload project and run full training on T4 GPU.

---

## Patent and Journal Publication Strategy

### Before Month 1 (Immediately)
- [ ] Do patent prior art search on `ipindia.gov.in`.
- [ ] Talk to college IP cell / research coordinator.
- [ ] Prepare provisional patent application (architecture block diagram + description).
- [ ] File provisional patent (INR 1,750).
- [ ] **Do NOT publish anything yet.**

### During Months 1-3 (Parallel Activity)
- [ ] Write Introduction + Literature Review for the paper.
- [ ] Keep detailed training logs (screenshots, CSV exports, accuracy/loss curves).
- [ ] Prepare complete patent specification.
- [ ] Submit complete patent within 12 months of provisional filing.

### End of Month 3
- [ ] Complete the paper draft (8 pages, IEEE format).
- [ ] Run Turnitin / plagiarism check (< 15%).
- [ ] Run Grammarly.
- [ ] Submit to IETE Journal of Research (1st choice).
- [ ] Submit simultaneously to DSJ (Defence Science Journal) or INDJCST as backup option.

### After Submission
- [ ] Present at NCC or INDICON (national conference).
- [ ] Respond to reviewer comments within the deadline.
- [ ] Update patent claims if architecture changes during review.
- [ ] Add conference paper and journal paper to your resume.

### Paper Requirements Checklist (Must Have)
- [ ] Clear problem statement (1 paragraph, crisp).
- [ ] Literature review mentioning Xu-Net and Yedroudj-Net and limitations.
- [ ] Architecture diagram (the 3-branch CNN diagram).
- [ ] Dataset description (BOSS, how you created stego pairs).
- [ ] Evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC/AUC, Confusion matrix).
- [ ] Comparison Table (Your model vs Xu-Net vs Yedroudj-Net).
- [ ] Ablation study (Show value of each branch).
- [ ] Quantization results (Model size and inference time on CPU).
