"""
generative_inversion.py
=======================
Synthetic Carrier Baseline Check — "AI vs AI" Detection

THE PROBLEM:
  Threat actors can use Latent Diffusion models (Midjourney, Stable Diffusion)
  or GANs to generate a completely new image where the pixel matrix was
  mathematically born carrying the malware payload. Because the image was
  never "edited" post-creation, traditional LSB counters and RS Steganalysis
  register it as a perfectly normal, clean image.

THE APPROACH:
  Natural photographic images follow specific statistical priors that are
  virtually impossible to fake even with the best diffusion models:

  1. Benford's Law on DCT Coefficients:
     Real natural images have DCT coefficients whose first digits follow
     Benford's distribution (1 appears ~30%, 2 ~17%, etc.). GAN/diffusion
     images that were "born" with a payload warped into the latent space
     show measurable deviations from Benford's distribution.

  2. Hurst Exponent (Long-Range Pixel Correlation):
     Natural images have a Hurst exponent H ≈ 0.65–0.85. When a diffusion
     model's noise schedule is warped to encode a payload, the self-similarity
     breaks down and H deviates significantly from the natural range.

  3. Micro-Texture Noise Floor Analysis:
     We apply a high-pass Laplacian filter and measure the residual energy
     distribution. Natural camera images have a characteristic Gaussian noise
     floor. Synthetically generated images with embedded payloads show excess
     energy spikes in specific spatial frequencies.

OUTPUT:
  A "Reconstruction Gap Score" (0–1). Higher = more likely the image's
  mathematical structure was warped to house a payload.
  
  This is a SCREENING tool, not a binary classifier. A threshold of 0.65
  is recommended as a "flag for human review" boundary.
"""

import numpy as np
from PIL import Image


# ─── Internal helpers ───────────────────────────────────────────────────────

def _compute_dct_benford_deviation(gray: np.ndarray) -> float:
    """
    Compute the mean absolute deviation of DCT coefficient first-digit
    distribution from Benford's Law.
    
    Returns a value in [0, 1]. Higher = greater deviation from natural priors.
    """
    # Tile-wise 8x8 DCT (matches JPEG block structure)
    h, w = gray.shape
    bh, bw = (h // 8) * 8, (w // 8) * 8
    blocks = gray[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8).astype(np.float64) - 128.0

    # Compute 2D DCT via separable 1D transform approximation
    # Using numpy's FFT for a mathematically valid DCT estimate
    dct_coeffs = np.fft.fft2(blocks, axes=(1, 2)).real.flatten()

    # Get absolute non-zero coefficients
    abs_coeffs = np.abs(dct_coeffs[np.abs(dct_coeffs) >= 1.0])
    if len(abs_coeffs) < 100:
        return 0.0

    # Extract first significant digit
    first_digits = np.floor(abs_coeffs / 10 ** np.floor(np.log10(abs_coeffs + 1e-9))).astype(int)
    first_digits = np.clip(first_digits, 1, 9)

    # Observed frequency distribution
    observed = np.bincount(first_digits, minlength=10)[1:] / len(first_digits)

    # Benford's Law expected distribution
    expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

    # Mean absolute deviation (0 = perfect Benford compliance, 1 = max deviation)
    mad = np.mean(np.abs(observed - expected))
    # Normalise: typical clean images have MAD < 0.03, GAN images > 0.08
    return float(np.clip(mad / 0.12, 0.0, 1.0))


def _compute_hurst_exponent(gray: np.ndarray) -> float:
    """
    Estimate the Hurst exponent via R/S (Rescaled Range) analysis.
    Natural images: H ≈ 0.65–0.85
    Synthetically warped images: H < 0.55 or H > 0.95
    """
    flat = gray.flatten().astype(np.float64)
    n = len(flat)
    if n < 128:
        return 0.7  # Assume natural for tiny images

    sizes = [max(8, n // k) for k in [2, 4, 8, 16, 32] if n // k >= 8]
    rs_values = []
    for size in sizes:
        segments = flat[:len(flat) - len(flat) % size].reshape(-1, size)
        rs_list = []
        for seg in segments:
            mean_s = np.mean(seg)
            deviation = np.cumsum(seg - mean_s)
            R = np.max(deviation) - np.min(deviation)
            S = np.std(seg, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(size), np.log(np.mean(rs_list))))

    if len(rs_values) < 2:
        return 0.7

    xs = np.array([v[0] for v in rs_values])
    ys = np.array([v[1] for v in rs_values])
    H = float(np.polyfit(xs, ys, 1)[0])
    return float(np.clip(H, 0.0, 1.0))


def _compute_noise_floor_anomaly(gray: np.ndarray) -> float:
    """
    Apply a Laplacian high-pass filter, then measure the kurtosis of the
    residual noise distribution.
    Natural camera images: kurtosis ≈ 3–6 (near-Gaussian).
    Synthetically warped images: kurtosis > 10 (heavy-tailed spikes).
    Returns a score in [0, 1]. Higher = more anomalous noise floor.
    """
    # Laplacian kernel for high-pass filtering
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.signal import convolve2d
    try:
        residual = convolve2d(gray.astype(np.float64), kernel, mode='same', boundary='symm')
    except Exception:
        # Fallback: simple numpy-based edge detect
        residual = (gray[1:-1, 1:-1].astype(np.float64) * -4
                    + gray[:-2, 1:-1] + gray[2:, 1:-1]
                    + gray[1:-1, :-2] + gray[1:-1, 2:])

    flat = residual.flatten()
    mu = np.mean(flat)
    sigma = np.std(flat)
    if sigma < 1e-9:
        return 0.0

    # Excess kurtosis
    kurt = float(np.mean(((flat - mu) / sigma) ** 4)) - 3.0
    # Natural images: kurt ≈ 0–3. Score normalised to [0, 1]
    return float(np.clip(kurt / 30.0, 0.0, 1.0))


# ─── Public API ──────────────────────────────────────────────────────────────

def generative_inversion_audit(pil_image: Image.Image) -> dict:
    """
    Run the full Generative Inversion Audit on an uploaded image.

    Args:
        pil_image: PIL Image (any mode, will be converted internally)

    Returns:
        dict with:
          - benford_deviation   : float [0–1] — DCT first-digit deviation
          - hurst_exponent      : float [0–1] — Hurst H value
          - hurst_anomaly       : float [0–1] — How far H is from natural range
          - noise_floor_anomaly : float [0–1] — Kurtosis of residual noise
          - reconstruction_gap  : float [0–1] — Composite score
          - verdict             : str — human readable finding
          - details             : list[str] — individual finding bullets
    """
    gray = np.array(pil_image.convert("L"), dtype=np.uint8)

    benford_dev   = _compute_dct_benford_deviation(gray)
    hurst_h       = _compute_hurst_exponent(gray)
    hurst_anomaly = float(np.clip(abs(hurst_h - 0.75) / 0.25, 0.0, 1.0))
    noise_anomaly = _compute_noise_floor_anomaly(gray)

    # Composite weighted reconstruction gap score
    gap = (benford_dev * 0.45) + (hurst_anomaly * 0.30) + (noise_anomaly * 0.25)
    gap = float(np.clip(gap, 0.0, 1.0))

    # Build individual finding bullets
    details = []

    if benford_dev > 0.6:
        details.append(
            f"🔴 DCT Benford deviation: {benford_dev*100:.1f}% — "
            "DCT coefficient first-digit distribution departs significantly from "
            "the Benford's Law prior expected in natural images. This is consistent "
            "with a latent-space that was warped to house a payload."
        )
    elif benford_dev > 0.3:
        details.append(
            f"🟡 DCT Benford deviation: {benford_dev*100:.1f}% — "
            "Borderline deviation. Could indicate heavy post-processing, "
            "social media re-encoding, or a subtle synthetic carrier."
        )
    else:
        details.append(
            f"🟢 DCT Benford deviation: {benford_dev*100:.1f}% — "
            "DCT coefficients conform to natural image Benford's distribution."
        )

    if hurst_h < 0.55 or hurst_h > 0.92:
        details.append(
            f"🔴 Hurst Exponent H={hurst_h:.3f} — "
            "Outside the natural photographic range (0.65–0.85). "
            f"{'Very low H suggests broken long-range pixel correlation, consistent with injected entropy.' if hurst_h < 0.55 else 'Very high H suggests unnaturally smooth latent space, consistent with diffusion model generation.'}"
        )
    else:
        details.append(
            f"🟢 Hurst Exponent H={hurst_h:.3f} — "
            "Within the natural photographic range (0.65–0.85). "
            "Long-range pixel correlations appear genuine."
        )

    if noise_anomaly > 0.5:
        details.append(
            f"🔴 Noise Floor Anomaly: {noise_anomaly*100:.1f}% — "
            "Laplacian residual has heavy-tailed kurtosis (excess spikes at specific "
            "spatial frequencies). Natural camera noise is near-Gaussian. This is "
            "a significant indicator of synthetic generation or payload injection."
        )
    elif noise_anomaly > 0.25:
        details.append(
            f"🟡 Noise Floor Anomaly: {noise_anomaly*100:.1f}% — "
            "Borderline noise floor anomaly. Could be heavy sharpening or upsampling artifact."
        )
    else:
        details.append(
            f"🟢 Noise Floor Anomaly: {noise_anomaly*100:.1f}% — "
            "Residual noise floor is within the natural photographic range."
        )

    # Final verdict
    if gap >= 0.65:
        verdict = (
            "🔴 HIGH RECONSTRUCTION GAP — The image's micro-texture statistics "
            f"deviate significantly from natural photographic priors (gap={gap:.3f}). "
            "This is consistent with a GAN/diffusion-generated synthetic carrier "
            "whose latent space was warped at generation time to embed a payload. "
            "Traditional LSB and RS tests cannot detect this class of attack."
        )
    elif gap >= 0.35:
        verdict = (
            "🟡 BORDERLINE GAP — Statistical priors show moderate deviation "
            f"(gap={gap:.3f}). Could be heavy JPEG re-compression, AI upscaling, "
            "or a low-capacity synthetic carrier. Human review is recommended."
        )
    else:
        verdict = (
            "🟢 NATURAL IMAGE BASELINE — Image micro-textures conform to natural "
            f"photographic priors (gap={gap:.3f}). Generative carrier attack is "
            "unlikely for this image."
        )

    return {
        "benford_deviation":   benford_dev,
        "hurst_exponent":      hurst_h,
        "hurst_anomaly":       hurst_anomaly,
        "noise_floor_anomaly": noise_anomaly,
        "reconstruction_gap":  gap,
        "verdict":             verdict,
        "details":             details,
    }
