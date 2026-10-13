# DRDO & ISRO Viva: Strategic Action Plan & Upgrades
*Prepared: September 2026 for StegShield / Sudarshana*

## 1. Core Viva Defense Strategy (How to impress DRDO/ISRO)
When defending this project to top-tier defense scientists, **do not bluff**. Focus on these three genuine architectural novelties:
1. **Cross-Domain Fusion:** You didn't just use an off-the-shelf CNN. You built a 3-branch network that simultaneously analyzes spatial noise (SRM), frequency shifts (DCT), and high-order statistics, fusing them via L2-normalization.
2. **Edge-Deployment Optimization:** Defense hardware (like UAVs and satellites) use edge devices like NVIDIA Jetsons, not heavy cloud GPUs. By applying **INT8 Quantization**, you shrunk the model to 1.1 MB, allowing real-time, low-latency offline inference.
3. **Acknowledge Limitations:** If asked about Diffusion Models / Coverless Generative Steganography, honestly state that this is currently a "Mode A" (Modified-Cover) detector. This shows immense maturity.

---

## 2. Hardware & Training Logistics
**The "Checkpoint Resuming" Strategy:**
*   **Phase 1 (Prototyping & Grind):** Use the free **Google Colab (Tesla T4)** to train the initial weights up to ~90% accuracy. Save the `fusion_best.pt` checkpoint file.
*   **Phase 2 (The Sprint):** Rent a powerful cloud GPU (like an RTX A6000 on RunPod) for a few hours. Upload your `.pt` checkpoint and *resume* training. The massive VRAM will allow you to increase the batch size and squeeze out the final 95-98% accuracy for only ₹300-₹500.
*   **Alternative:** Your **Lenovo Yoga Book (RTX 5060, 1TB SSD, 16GB RAM)** is an absolute beast for this. It has faster disk I/O than Colab and won't randomly disconnect. If you use it, set the `batch_size` to 8 (due to 8GB VRAM), enable Mixed Precision (`autocast`), and use "Tent Mode" with a cooling pad to prevent thermal throttling over a 30-hour run.

---

## 3. The 3-Month Future Upgrade Plan
With 3 months remaining before the interview, we will upgrade this project into a true defense-grade prototype. We will implement these in order:

### Upgrade A: Explainable AI (Grad-CAM Heatmaps)
*   **What:** We will modify the PyTorch code to generate a visual heatmap over the analyzed image.
*   **Why:** DRDO hates "Black Box" AI. The heatmap will prove *why* the model flagged an image by highlighting the exact pixels containing steganographic noise.

### Upgrade B: Generative Steganalysis (Branch D)
*   **What:** Add a 4th branch dedicated to detecting AI-generated images (Coverless Steganography).
*   **How:** We will download a HuggingFace dataset of Real vs. AI-Generated images and use **Transfer Learning** on a pre-trained Vision Transformer (ViT) to fine-tune it in just a few epochs.

### Upgrade C: Transformer Cross-Attention Fusion
*   **What:** Replace the standard Dense (Linear) layers in `fusion_model.py` with a Cross-Attention mechanism.
*   **Why:** This allows the network to dynamically shift its "attention" between the Spatial and Frequency branches depending on the attack type, making the architecture mathematically state-of-the-art.

### Upgrade D: Military/Enterprise Packaging
*   **What:** Write a `Dockerfile` and a `FastAPI` backend so the model can be deployed as an industry-standard microservice, just like actual defense contractors do.
