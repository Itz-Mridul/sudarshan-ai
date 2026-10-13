# 🚀 Google Colab Training Guide — Sudarshana (StegShield) 4-Branch System

This guide takes you step-by-step through training **Branch D (Generative AI Detector)** and the **4-Branch Fusion System** on Google Colab with a free T4 GPU.

---

## 📋 Pre-Flight Checklist

1. Google Account with Google Drive access (~4-5 GB free space).
2. Cleaned local project (already completed: ~8.9 GB junk removed, 26/26 unit tests passing).
3. Google Colab running Python 3 with **T4 GPU** enabled.

---

## 📦 Step 1: Prepare the Lightweight Colab Zip (One Command)

To upload to Google Drive without slow per-file uploading, create a compressed zip containing only the necessary data (`data/clean` + `data/stego`) and code (excluding `.git`, `external_repos`, and duplicate `BOSSbase_1.01`):

Run this terminal command from inside `EDI_Project`:
```bash
zip -r ../EDI_Project_Colab.zip . \
  -x ".git/*" \
  -x "data/BOSSbase_1.01/*" \
  -x "external_repos/*" \
  -x "**/__pycache__/*" \
  -x "*.DS_Store"
```
*This creates `EDI_Project_Colab.zip` in your parent directory, saving over 3 GB of redundant upload time.*

Upload `EDI_Project_Colab.zip` to the **root of your Google Drive** (`My Drive/`).

---

## 💻 Step 2: Open Google Colab & Select T4 GPU

1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **File → Upload notebook** and select `notebooks/StegShield_Training.ipynb` (or create a blank Python 3 notebook and copy the cells from `notebooks/colab_cells.txt`).
3. Set the hardware accelerator:
   - Click **Runtime → Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU**.
   - Click **Save**.

---

## 🏃 Step 3: Run the Training Cells in Order

### Cell 1: Mount Drive & Extract Project to Fast NVMe SSD
```python
from google.colab import drive
import os, sys, shutil

drive.mount('/content/drive')

# Extract zip to fast local Colab NVMe SSD
if os.path.exists('/content/drive/MyDrive/EDI_Project_Colab.zip'):
    print("📦 Extracting project to local Colab NVMe SSD...")
    !unzip -q /content/drive/MyDrive/EDI_Project_Colab.zip -d /content/EDI_Project
    os.chdir('/content/EDI_Project')
elif os.path.exists('/content/drive/MyDrive/EDI_Project'):
    os.chdir('/content/drive/MyDrive/EDI_Project')

print("Current Directory:", os.getcwd())
sys.path.insert(0, 'src')
```

### Cell 2: Install Requirements & Verify GPU
```python
!pip install -r requirements.txt -q
!pip install timm -q

import torch
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ No GPU detected! Go to: Runtime > Change runtime type > T4 GPU")
```

### Cell 3: Quick Sanity Test (~1 minute)
```python
!python src/training/train.py --mode branch_d --max_images 100 --epochs 2
```

### Cell 4: Train Branch D — Generative AI Stego Detector (~30-45 mins)
Fine-tunes EfficientNet to detect diffusion- and GAN-based coverless steganography:
```python
!python src/training/train.py --mode branch_d --epochs 40 --lr 1e-4
```
*Expected validation accuracy: 82%–88%*

### Cell 5: Train 4-Branch Fusion Head (~30-40 mins)
Loads pretrained branches A (Pixel/SRM), B (DCT Frequency), C (Stats MLP), and D (Generative) with frozen weights and trains the fusion head:
```python
!python src/training/train.py --mode fusion --epochs 40 --lr 1e-3
```
*Expected validation accuracy: 86%–90%*

### Cell 6: End-to-End Joint Fine-Tuning (~45-60 mins)
Unfreezes all 4 branches at a low learning rate (`1e-5`) to avoid catastrophic forgetting and reach peak performance:
```python
!python src/training/train.py --mode joint --epochs 25 --lr 1e-5
```
*Target validation accuracy: 92%–95%*

---

## 📊 Step 4: Run Evaluation & Generate IEEE Paper Figures

### Cell 7: Generate Metrics, Confusion Matrix, ROC Curves & Ablation
```python
!python src/training/evaluate.py --checkpoint weights/fusion_best.pt --mode fusion
!python generate_paper_figures.py
!python ablation_study.py --fusion_ckpt weights/fusion_best.pt
```

### Cell 8: INT8 Model Quantization (For Edge/Defense Deployment)
```python
!python src/deploy/quantize.py --model_path weights/fusion_best.pt --output_path weights/fusion_int8.pt
```

---

## 💾 Step 5: Save & Download Trained Models

### Cell 9: Backup Weights and Results to Google Drive
```python
os.makedirs('/content/drive/MyDrive/EDI_Trained_Output', exist_ok=True)
shutil.make_archive('/content/drive/MyDrive/EDI_Trained_Output/weights_trained', 'zip', 'weights/')
shutil.make_archive('/content/drive/MyDrive/EDI_Trained_Output/results_trained', 'zip', 'results/')
print("🎉 Trained weights and results saved to 'EDI_Trained_Output' in Google Drive!")
```

### Download Back to Mac:
Download `weights_trained.zip` and `results_trained.zip` from Google Drive, unzip them, and place them into your local `weights/` and `results/` folders.
