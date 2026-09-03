# Steganalysis Research Papers — Key Summaries

Based on an online literature review of the foundational papers listed in Section 3 of your PRD, here are the detailed summaries of their core contributions, architectures, and relevance to your Multi-Branch CNN project.

## 1. Xu-Net (2016)
**"Structural Design of Convolutional Neural Networks for Steganalysis" (Xu et al.)**
- **Relevance:** This is your primary baseline model.
- **Core Innovation:** It was one of the first CNNs specifically tailored for image steganalysis rather than general computer vision. It highlighted that traditional CV architectures (like VGG or ResNet) perform poorly on steganographic noise.
- **Architecture Highlights:** 
  - **ABS Layer:** Takes the absolute values of elements in the feature maps generated from the first convolutional layer (which applies high-pass filters) to improve statistical modeling.
  - **TanH Activation:** Uses hyperbolic tangent (TanH) in early stages to prevent overfitting by constraining the range of data values.
  - **1x1 Convolutions:** Used in deeper layers to reduce the strength of modeling and dimensionality.
- **Limitation:** It is a pixel-only analysis model, meaning it struggles to detect frequency-domain hiding (like JPEG DCT manipulation).

## 2. Yedroudj-Net (2018)
**"Yedroudj-Net: An efficient CNN for spatial steganalysis" (Yedroudj et al.)**
- **Relevance:** Your secondary baseline.
- **Core Innovation:** Designed to improve classification accuracy and training stability in the spatial domain. It merges the best practices from rich models and earlier CNNs (like Xu-Net).
- **Architecture Highlights:** Uses a 7-block CNN with average-pooling layers to effectively capture spatial stego noise. It serves as a strong architectural reference for your **Branch A** convolution blocks.
- **Limitation:** Fails or degrades significantly when the image is subjected to basic transformations like resizing or JPEG compression after the steganography is applied.

## 3. SRNet (2019)
**"Deep Residual Network for Steganalysis of Digital Images" (Boroumand et al.)**
- **Relevance:** The gold standard for deep residual steganalysis.
- **Core Innovation:** Unlike prior models (including Xu-Net) that relied on fixed, pre-computed high-pass filters (like SRM) in the first layer to suppress image content, SRNet learns these noise residuals *automatically* through deep residual learning.
- **Architecture Highlights:** Uses unpooled residual blocks in the early layers to extract noise residuals, followed by spatial pooling layers.
- **Performance:** Achieves extremely high accuracy (~89% on S-UNIWARD 0.4bpp).
- **Limitation:** Highly GPU-dependent and computationally heavy, making it unsuitable for your air-gapped CPU-only deployment constraint.

## 4. GNCNN (2015)
**"A New CNN Design for Image Steganalysis" (Qian et al.)**
- **Relevance:** The pioneer of applying CNNs to steganalysis.
- **Core Innovation:** Replaced standard activation functions (like ReLU) with a **Gaussian activation function** in the convolutional layers. This was hypothesized to better capture the very weak, zero-mean signals typical of steganographic modifications.
- **Architecture Highlights:** Uses fixed KV filters for preprocessing, followed by 5 convolutional layers and a fully connected classifier.
- **Limitation:** Performance drops sharply at low payload densities (e.g., 0.1 bpp) and is generally superseded by Xu-Net and SRNet.

## 5. Depth-Wise Separable CNN (Zhang et al., 2020/2022)
**"Depth-Wise Separable Convolutions and Multi-Level Pooling for an Efficient Spatial CNN-Based Steganalysis"**
- **Relevance:** Crucial for your INT8 quantization / CPU deployment goals.
- **Core Innovation:** Aimed at reducing model parameters while maintaining high accuracy, making the model more lightweight.
- **Architecture Highlights:** 
  - **3x3 Kernels:** Replaced traditional 5x5 kernels in the preprocessing layer to reduce parameters.
  - **Depth-Wise Separable Convolutions:** Exploits channel correlations of image residuals while heavily compressing the model size.
  - **Spatial Pyramid Pooling (SPP):** Aggregates multi-level features.
- **Takeaway for you:** The parameter-efficiency techniques discussed here justify why your multi-branch model can still run efficiently on a CPU after INT8 quantization.

---
> [!NOTE] 
> The recent 2024/2025 papers (Li et al., Uspenskyi, PENet+) build on these foundations by incorporating Transformers, Self-Supervised Learning (SSL), and further lightweight optimizations. Their core takeaway is that **generalizability across different hiding methods** remains the biggest unsolved problem, heavily reinforcing your PRD's research gap of using a Multi-Branch fusion approach.
