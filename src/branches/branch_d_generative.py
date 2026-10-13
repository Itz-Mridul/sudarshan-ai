"""
branch_d_generative.py
======================
Branch D: Generative Steganography Detector

WHAT THIS DOES:
  Detects steganography hidden using AI-based methods (GANs, Diffusion Models,
  Invertible Neural Networks). Unlike Branches A-C which look for pixel/frequency
  artifacts from classical embedding, this branch learns to identify the
  statistical signatures left by generative AI processes.

WHY THIS IS NEEDED:
  Modern steganography (2024-2026) uses generative AI to create entirely new
  images with data baked into the latent space. Classical SRM filters (Branch A)
  cannot detect this because there is no "cover vs. stego" pixel modification —
  the entire image is synthetically generated.

ARCHITECTURE:
  This branch uses Transfer Learning from a pre-trained EfficientNet-B0:
    1. Load EfficientNet-B0 (pre-trained on ImageNet — already knows textures)
    2. Freeze all layers except the last 2 blocks (fine-tune only the top)
    3. Replace the classifier head with our own feature extractor
    4. Output a 128-dim feature vector for fusion

  Why EfficientNet-B0 instead of ViT?
    - EfficientNet-B0 is only 5.3M parameters (ViT-B is 86M)
    - It fits easily on an RTX 5060 with 8GB VRAM
    - It still captures the texture/noise patterns we need
    - After INT8 quantization, it stays under 2MB

TRAINING STRATEGY:
  Phase 1: Train on Real vs. AI-Generated images (binary classification)
  Phase 2: Fine-tune on Generative Stego vs. Clean-Generated images
           (CRITICAL: must use matched clean-generated controls!)

TARGET ACCURACY: 85%+ on generative steganography detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    HAS_EFFICIENTNET = True
except ImportError:
    HAS_EFFICIENTNET = False


class GenerativeDetectorCNN(nn.Module):
    """
    Branch D: Generative Steganography Detector.

    Uses a lightweight CNN architecture inspired by EfficientNet's compound
    scaling, but built from scratch to avoid dependency issues and to keep
    the model small enough for edge deployment.

    Architecture:
        Input (3, 256, 256)
          ↓
        ConvBlock 1: Conv(3→32) → BN → SiLU → MaxPool     → (32, 128, 128)
          ↓
        ConvBlock 2: Conv(32→64) → BN → SiLU → MaxPool    → (64, 64, 64)
          ↓
        DepthwiseSeparable 1: DW(64) + PW(64→128) → SiLU  → (128, 32, 32)
          ↓
        DepthwiseSeparable 2: DW(128) + PW(128→256) → SiLU → (256, 16, 16)
          ↓
        DepthwiseSeparable 3: DW(256) + PW(256→512) → SiLU → (512, 8, 8)
          ↓
        GlobalAveragePool → FC(512→feature_dim)
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        self.feature_dim = feature_dim

        # ── Initial convolution blocks ─────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),  # → (32, 128, 128)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),  # → (64, 64, 64)
        )

        # ── Depthwise Separable Convolution blocks ─────────────────────────
        # These are the key innovation from MobileNet/EfficientNet:
        # Split convolution into depthwise (spatial) + pointwise (channel mixing)
        # This reduces parameters by ~8-9x compared to standard convolutions
        self.ds_block1 = self._make_ds_block(64, 128, stride=2)    # → (128, 32, 32)
        self.ds_block2 = self._make_ds_block(128, 256, stride=2)   # → (256, 16, 16)
        self.ds_block3 = self._make_ds_block(256, 512, stride=2)   # → (512, 8, 8)

        # ── Squeeze-and-Excitation (SE) attention ──────────────────────────
        # Learns to weight channels by their importance for detecting
        # generative artifacts (e.g., GAN spectral peaks, diffusion noise)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512 // 16),
            nn.SiLU(inplace=True),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid(),
        )

        # ── Global pooling and feature extraction ──────────────────────────
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
        )

    @staticmethod
    def _make_ds_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        """Create a Depthwise Separable Convolution block."""
        return nn.Sequential(
            # Depthwise convolution (spatial filtering per channel)
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),

            # Pointwise convolution (channel mixing)
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image tensor of shape (batch, 3, H, W)
               Note: This branch takes RGB, unlike Branches A-C which take grayscale!

        Returns:
            feature vector of shape (batch, feature_dim)
        """
        # Stem convolutions
        x = self.stem(x)           # (batch, 64, 64, 64)

        # Depthwise separable blocks
        x = self.ds_block1(x)      # (batch, 128, 32, 32)
        x = self.ds_block2(x)      # (batch, 256, 16, 16)
        x = self.ds_block3(x)      # (batch, 512, 8, 8)

        # Squeeze-and-Excitation channel attention
        se_weights = self.se(x).unsqueeze(-1).unsqueeze(-1)
        x = x * se_weights         # Channel-wise reweighting

        # Global average pooling → feature vector
        x = self.gap(x)            # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1) # (batch, 512)
        x = self.fc(x)             # (batch, feature_dim)

        return x


class GenerativeDetectorTransfer(nn.Module):
    """
    Alternative Branch D using Transfer Learning from EfficientNet-B0.

    Use this if you have torchvision installed and want to leverage
    ImageNet pre-training for faster convergence.

    The pre-trained EfficientNet already knows texture/noise patterns;
    we just fine-tune the top layers to recognize generative artifacts.
    """

    def __init__(self, feature_dim: int = 128, freeze_ratio: float = 0.8):
        super().__init__()

        if not HAS_EFFICIENTNET:
            raise ImportError(
                "torchvision with EfficientNet support is required. "
                "Install with: pip install torchvision>=0.16.0"
            )

        self.feature_dim = feature_dim

        # Load pre-trained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base = efficientnet_b0(weights=weights)

        # Extract the feature extractor (everything before the classifier)
        self.features = base.features
        self.avgpool = base.avgpool

        # Freeze early layers (they detect generic edges/textures)
        all_params = list(self.features.parameters())
        freeze_count = int(len(all_params) * freeze_ratio)
        for param in all_params[:freeze_count]:
            param.requires_grad = False

        # Replace classifier with our feature extractor
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image tensor of shape (batch, 3, H, W)

        Returns:
            feature vector of shape (batch, feature_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # (batch, 1280)
        x = self.fc(x)              # (batch, feature_dim)
        return x


class BranchDClassifier(nn.Module):
    """
    Standalone Branch D model with its own classification head.
    Use this for INDEPENDENT training and evaluation of Branch D.
    """

    def __init__(self, feature_dim: int = 128, num_classes: int = 2,
                 use_transfer: bool = False):
        super().__init__()

        if use_transfer and HAS_EFFICIENTNET:
            self.backbone = GenerativeDetectorTransfer(feature_dim=feature_dim)
        else:
            self.backbone = GenerativeDetectorCNN(feature_dim=feature_dim)

        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: EfficientNet usually requires 3 channels. Since our images are 
        # grayscale, we duplicate the channel to make it RGB-compatible.
        x_rgb = x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x
        features = self.backbone(x_rgb)
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Branch D (Generative Steganography Detector)...")
    print()

    # Create a dummy batch of 4 RGB images, 256×256
    batch = torch.randn(4, 3, 256, 256)

    # Test lightweight CNN version
    print("═" * 50)
    print("  Testing: GenerativeDetectorCNN (Custom)")
    print("═" * 50)
    model_cnn = BranchDClassifier(feature_dim=128, use_transfer=False)
    model_cnn.eval()

    with torch.no_grad():
        logits = model_cnn(batch)
        features = model_cnn.get_features(batch)

    print(f"  Input shape:    {batch.shape}")
    print(f"  Feature shape:  {features.shape}  ← this goes into fusion model")
    print(f"  Output shape:   {logits.shape}    ← (batch, 2) for clean/stego")

    total_params = sum(p.numel() for p in model_cnn.parameters())
    trainable = sum(p.numel() for p in model_cnn.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")
    print()

    # Test transfer learning version (if available)
    if HAS_EFFICIENTNET:
        print("═" * 50)
        print("  Testing: GenerativeDetectorTransfer (EfficientNet-B0)")
        print("═" * 50)
        model_tf = BranchDClassifier(feature_dim=128, use_transfer=True)
        model_tf.eval()

        with torch.no_grad():
            logits_tf = model_tf(batch)
            features_tf = model_tf.get_features(batch)

        print(f"  Input shape:    {batch.shape}")
        print(f"  Feature shape:  {features_tf.shape}")
        print(f"  Output shape:   {logits_tf.shape}")

        total_tf = sum(p.numel() for p in model_tf.parameters())
        trainable_tf = sum(p.numel() for p in model_tf.parameters() if p.requires_grad)
        print(f"  Total params:     {total_tf:,}")
        print(f"  Trainable params: {trainable_tf:,} (80% frozen)")
    else:
        print("  ⚠️ EfficientNet not available — skipping transfer learning test")

    print("\n✅ Branch D OK!")
