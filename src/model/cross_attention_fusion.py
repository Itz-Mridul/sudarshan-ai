"""
cross_attention_fusion.py
=========================
Cross-Attention Transformer Fusion Layer for Sudarshana

WHAT THIS DOES:
  Replaces the simple Dense (Linear) fusion head with a Transformer-based
  Cross-Attention mechanism. Instead of blindly concatenating branch features,
  this module allows each branch to dynamically "attend" to the other branches.

WHY THIS IS NOVEL:
  Traditional steganalysis models (SRNet, Xu-Net, Yedroudj-Net) use single-domain
  features. Even multi-branch models that exist simply concatenate features.
  Cross-Attention allows the Pixel branch to "ask" the Frequency branch:
  "Did you also see something suspicious here?" — and vice versa.

  This is the same mathematical mechanism used in modern Transformers (GPT, ViT),
  applied specifically to the steganalysis fusion problem.

ARCHITECTURE:
  Branch A features (256-dim) ─┐
  Branch B features (256-dim) ─┼─ Project to d_model ─ Cross-Attention ─ Fusion
  Branch C features  (32-dim) ─┘
                                     ↓
                              Dense(d_model → 2) → Softmax
                                     ↓
                          [CLEAN / STEGO + confidence %]

MATHEMATICAL FOUNDATION:
  Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

  Where:
    Q (Query)  = "What am I looking for?"
    K (Key)    = "What information do I have?"
    V (Value)  = "What should I return if matched?"

  Each branch acts as both a Query and a Key/Value source, enabling
  bidirectional information flow between spatial and frequency domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BranchProjection(nn.Module):
    """
    Projects a branch's feature vector to a common embedding dimension (d_model).
    This is necessary because Branch A (256-dim), Branch B (256-dim), and
    Branch C (32-dim) have different sizes. Cross-Attention requires all
    inputs to share the same dimensionality.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),  # GELU is the standard activation in Transformers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionBlock(nn.Module):
    """
    A single Cross-Attention block.

    Given a query branch and a set of key/value branches, computes
    multi-head attention to allow the query to selectively attend to
    information from the other branches.

    This implements:
      output = MultiHeadAttention(Q=query, K=context, V=context)
      output = LayerNorm(output + query)       # residual connection
      output = LayerNorm(FFN(output) + output) # feed-forward + residual
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Multi-head attention projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Layer norms and feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:   (batch, d_model) — the branch asking the question
            context: (batch, num_branches, d_model) — branches providing context

        Returns:
            (batch, d_model) — updated query representation
        """
        B = query.shape[0]

        # Expand query to have a sequence dimension: (batch, 1, d_model)
        q = query.unsqueeze(1)

        # Project Q, K, V
        Q = self.W_q(q)           # (batch, 1, d_model)
        K = self.W_k(context)     # (batch, num_branches, d_model)
        V = self.W_v(context)     # (batch, num_branches, d_model)

        # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
        Q = Q.view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back: (batch, 1, d_model) → (batch, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        attn_output = self.W_o(attn_output).squeeze(1)

        # Residual connection + LayerNorm
        x = self.norm1(attn_output + query)

        # Feed-forward network + Residual connection + LayerNorm
        x = self.norm2(self.ffn(x) + x)

        return x


class CrossAttentionFusion(nn.Module):
    """
    The complete Cross-Attention Fusion module for Sudarshana.

    Each branch gets to "attend" to the other branches, producing an
    attention-enhanced representation. These enhanced representations
    are then combined for the final classification.

    Architecture:
      1. Project each branch to d_model dimensions
      2. For each branch, run Cross-Attention with all branches as context
      3. Combine attended features → classification head

    This replaces the simple FusionLayer (Dense concatenation) in fusion_model.py.
    """

    def __init__(
        self,
        dim_a: int = 256,
        dim_b: int = 256,
        dim_c: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
        num_classes: int = 2,
    ):
        """
        Args:
            dim_a:       Branch A feature dimension (Pixel CNN)
            dim_b:       Branch B feature dimension (DCT CNN)
            dim_c:       Branch C feature dimension (Stats MLP)
            d_model:     Common embedding dimension for Cross-Attention
            n_heads:     Number of attention heads
            n_layers:    Number of stacked Cross-Attention layers
            dropout:     Dropout rate
            num_classes: Output classes (2 = clean/stego)
        """
        super().__init__()

        # Project each branch to d_model
        self.proj_a = BranchProjection(dim_a, d_model)
        self.proj_b = BranchProjection(dim_b, d_model)
        self.proj_c = BranchProjection(dim_c, d_model)

        # Cross-Attention layers for each branch (stacked)
        self.attn_layers_a = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.attn_layers_b = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.attn_layers_c = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Classification head: takes 3 * d_model → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 2, num_classes),
        )

        # Store attention weights for explainability
        self.last_attn_weights = None

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        feat_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_a: (batch, dim_a) — Branch A features
            feat_b: (batch, dim_b) — Branch B features
            feat_c: (batch, dim_c) — Branch C features

        Returns:
            logits: (batch, num_classes)
        """
        # Project to common dimension
        a = self.proj_a(feat_a)   # (batch, d_model)
        b = self.proj_b(feat_b)   # (batch, d_model)
        c = self.proj_c(feat_c)   # (batch, d_model)

        # Stack all branches as context: (batch, 3, d_model)
        context = torch.stack([a, b, c], dim=1)

        # Apply Cross-Attention layers
        for attn_a, attn_b, attn_c in zip(
            self.attn_layers_a, self.attn_layers_b, self.attn_layers_c
        ):
            a = attn_a(a, context)
            b = attn_b(b, context)
            c = attn_c(c, context)

            # Update context with new representations
            context = torch.stack([a, b, c], dim=1)

        # Concatenate attended features and classify
        combined = torch.cat([a, b, c], dim=1)   # (batch, 3 * d_model)
        logits = self.classifier(combined)         # (batch, num_classes)

        return logits


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Cross-Attention Fusion Layer...")
    print()

    batch_size = 4

    # Simulate branch outputs
    feat_a = torch.randn(batch_size, 256)   # Branch A (Pixel CNN)
    feat_b = torch.randn(batch_size, 256)   # Branch B (DCT CNN)
    feat_c = torch.randn(batch_size, 32)    # Branch C (Stats MLP)

    # Create fusion module
    fusion = CrossAttentionFusion(
        dim_a=256, dim_b=256, dim_c=32,
        d_model=128, n_heads=4, n_layers=2,
        num_classes=2
    )

    # Forward pass
    logits = fusion(feat_a, feat_b, feat_c)
    probs = F.softmax(logits, dim=1)

    print(f"Branch A input:  {feat_a.shape}")
    print(f"Branch B input:  {feat_b.shape}")
    print(f"Branch C input:  {feat_c.shape}")
    print(f"Output logits:   {logits.shape}")
    print(f"Output probs:    {probs.shape}")
    print()

    for i in range(batch_size):
        pred = "STEGO" if probs[i, 1] > 0.5 else "CLEAN"
        print(f"  Sample {i}: {pred} (stego_prob={probs[i,1]:.3f})")

    # Parameter count
    total = sum(p.numel() for p in fusion.parameters())
    print(f"\nTotal parameters: {total:,}")
    print("✅ Cross-Attention Fusion OK!")
