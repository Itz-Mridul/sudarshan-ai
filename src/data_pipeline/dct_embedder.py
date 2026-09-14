"""
dct_embedder.py
===============
Frequency-domain steganography generator (Simplified F5 / Jsteg Simulator).

This module simulates the core artifacts of JPEG-domain steganography by:
1. Converting spatial pixels to 8x8 DCT blocks.
2. Quantizing coefficients using standard JPEG luminance matrix.
3. Embedding data ONLY into non-zero AC coefficients (leaving DC and 0s untouched).
4. Dequantizing and applying Inverse DCT to return spatial pixels.

Output MUST be saved as a lossless PNG to preserve the exact frequency artifacts.
"""

import os
import hashlib
import numpy as np
from scipy.fft import dct, idct
from PIL import Image


# Standard JPEG luminance quantization matrix (Q50)
# Scaled according to quality standard
Q_MATRIX = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99]
], dtype=np.float32)


def apply_2d_dct(block: np.ndarray) -> np.ndarray:
    """Apply 2D Discrete Cosine Transform to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_2d_idct(block: np.ndarray) -> np.ndarray:
    """Apply 2D Inverse Discrete Cosine Transform to an 8x8 block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def embed_dct(image_array: np.ndarray, payload: bytes, seed: int = 42) -> np.ndarray:
    """
    Simulates JSteg / F5 JPEG-domain steganography.
    
    Args:
        image_array : 2D numpy array of shape (H, W), dtype uint8.
        payload     : raw bytes to embed.
        seed        : random seed to optionally scramble embedding order.
        
    Returns:
        stego_img   : 2D numpy array of the same shape, dtype uint8.
    """
    if image_array.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")
        
    h, w = image_array.shape
    
    # Pad to multiple of 8
    h_pad = (h + 7) // 8 * 8
    w_pad = (w + 7) // 8 * 8
    img_padded = np.zeros((h_pad, w_pad), dtype=np.float32)
    
    # Shift pixels by -128 (standard JPEG centering)
    img_padded[:h, :w] = image_array.astype(np.float32) - 128.0
    
    stego_img = np.zeros_like(img_padded)
    
    # Convert payload to bitstream
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    bit_len = len(bits)
    bit_idx = 0
    
    # For consistent embedding spread (like UNIWARD/F5), we use a PRNG
    rng = np.random.default_rng(seed)
    
    # Process blocks
    for i in range(0, h_pad, 8):
        for j in range(0, w_pad, 8):
            block = img_padded[i:i+8, j:j+8]
            dct_block = apply_2d_dct(block)
            
            # Quantize
            q_block = np.round(dct_block / Q_MATRIX)
            
            if bit_idx < bit_len:
                # Embed in AC coefficients (skip DC at 0,0)
                # Randomize coefficient visit order for security simulation
                coords = [(r, c) for r in range(8) for c in range(8) if not (r == 0 and c == 0)]
                rng.shuffle(coords)
                
                for r, c in coords:
                    if bit_idx >= bit_len:
                        break
                        
                    coeff = int(q_block[r, c])
                    # Skip 0s (Jsteg/F5 standard to avoid destroying smooth areas)
                    if coeff != 0:
                        # Simple LSB substitution on the quantized coefficient
                        # Using bitwise ops on integer coeff
                        new_coeff = (coeff & ~1) | int(bits[bit_idx])
                        q_block[r, c] = new_coeff
                        bit_idx += 1
                        
            # Dequantize
            dct_recon = q_block * Q_MATRIX
            
            # IDCT
            stego_img[i:i+8, j:j+8] = apply_2d_idct(dct_recon)
            
    if bit_idx < bit_len:
        raise ValueError(f"Payload too large. Embedded {bit_idx}/{bit_len} bits. Use a larger image or smaller payload.")
        
    # Unshift pixels back to 0-255 range and clip
    stego_img = np.clip(stego_img + 128.0, 0, 255).astype(np.uint8)
    
    # Crop back to original dimensions
    return stego_img[:h, :w]


def create_dct_stego_image(clean_path: str, stego_path: str, payload_fraction: float = 0.1):
    """
    Reads a clean image, embeds random data in the DCT domain, and saves as PNG.
    
    Args:
        clean_path : Path to input clean image.
        stego_path : Path to save stego image (MUST be .png).
        payload_fraction : Fraction of non-zero AC coefficients to overwrite.
    """
    # Enforce PNG output to prevent second-pass JPEG destruction
    base, ext = os.path.splitext(stego_path)
    if ext.lower() in [".jpg", ".jpeg"]:
        print(f"[WARNING] DCT stego output cannot be JPEG. Changing to PNG: {stego_path}")
        stego_path = base + ".png"
        
    img = Image.open(clean_path).convert("L")
    img_array = np.array(img, dtype=np.uint8)
    
    # Deterministic seed from filename (stable pairing)
    stem = os.path.splitext(os.path.basename(clean_path))[0]
    sha_hash = hashlib.sha256(stem.encode()).hexdigest()
    seed = int(sha_hash[:8], 16)
    
    # Estimate capacity (roughly 10% of total pixels is safe max for non-zero AC)
    max_bytes = int((img_array.size * 0.1) / 8)
    payload_size = max(1, int(max_bytes * payload_fraction))
    
    # Generate deterministic random payload based on stem
    rng = np.random.default_rng(seed)
    payload = rng.bytes(payload_size)
    
    # Embed
    try:
        stego_array = embed_dct(img_array, payload, seed=seed)
        Image.fromarray(stego_array).save(stego_path)
    except ValueError as e:
        print(f"Skipping {clean_path}: {e}")
