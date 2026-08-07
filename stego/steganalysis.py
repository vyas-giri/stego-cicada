import numpy as np
import jpegio
from scipy.stats import chi2
from PIL import Image
import os

def _compute_chi2_vectorized(counts: np.ndarray) -> tuple[float, int, float]:
    """Computes Chi-Square statistic, degrees of freedom, and p-value from a 256-bin histogram."""
    even = counts[0::2].astype(np.float64)
    odd = counts[1::2].astype(np.float64)
    total = even + odd

    valid = total >= 10
    if not np.any(valid):
        return 0.0, 0, 0.0

    chi2_stat = np.sum(((even[valid] - odd[valid]) ** 2) / total[valid])
    dof = int(np.sum(valid))

    if dof == 0:
        return 0.0, 0, 0.0

    p_val = float(chi2.sf(chi2_stat, dof))
    return float(chi2_stat), dof, p_val

def analyze_png_lsb(image_path: str, num_chunks: int = 50) -> dict:
    """Fast vectorized Chi-Square steganalysis for PNG images using histogram prefix sums."""
    img = Image.open(image_path).convert('RGB')
    flat_pixels = np.array(img).flatten()

    # Global analysis in < 5ms
    global_counts = np.bincount(flat_pixels, minlength=256)
    global_chi2, global_dof, global_p = _compute_chi2_vectorized(global_counts)

    # Vectorized progressive chunked scanning via cumulative histogram sums
    chunk_size = max(1024, len(flat_pixels) // num_chunks)
    chunks = [flat_pixels[i:i + chunk_size] for i in range(0, len(flat_pixels), chunk_size)]
    
    # Pre-compute 2D histogram matrix (num_chunks x 256)
    chunk_histograms = np.array([np.bincount(c, minlength=256) for c in chunks])
    prefix_histograms = np.cumsum(chunk_histograms, axis=0)

    max_chunk_p = 0.0
    detected_chunk = False

    for hist in prefix_histograms:
        _, _, p_val = _compute_chi2_vectorized(hist)
        if p_val > max_chunk_p:
            max_chunk_p = p_val
        if p_val > 0.95:
            detected_chunk = True
            break

    # Stego is detected ONLY if pair equalization occurs (p > 0.95)
    stego_detected = (global_p > 0.95) or detected_chunk

    return {
        "stego_probability": max(global_p, max_chunk_p),
        "detected": stego_detected,
        "chi2_stat": global_chi2,
        "dof": global_dof
    }

def analyze_jpeg_dct(image_path: str, num_chunks: int = 50) -> dict:
    """Fast vectorized Chi-Square steganalysis for JPEG AC coefficients."""
    jpeg_obj = jpegio.read(image_path)
    y_coefs = jpeg_obj.coef_arrays[0]

    h, w = y_coefs.shape
    ac_mask = np.ones((h, w), dtype=bool)
    ac_mask[::8, ::8] = False  # Exclude DC

    nonzero_ac = y_coefs[ac_mask & (y_coefs != 0)]
    
    # Map signed int16 coefficients to positive indices for bincount
    shift = int(np.abs(nonzero_ac.min())) if len(nonzero_ac) > 0 and nonzero_ac.min() < 0 else 0
    shifted_ac = (nonzero_ac + shift).astype(np.int32)
    max_val = int(shifted_ac.max()) + 1 if len(shifted_ac) > 0 else 1

    global_counts = np.bincount(shifted_ac, minlength=max_val)
    global_chi2, global_dof, global_p = _compute_chi2_vectorized(global_counts)

    chunk_size = max(1024, len(shifted_ac) // num_chunks)
    chunks = [shifted_ac[i:i + chunk_size] for i in range(0, len(shifted_ac), chunk_size)]
    chunk_histograms = np.array([np.bincount(c, minlength=max_val) for c in chunks])
    prefix_histograms = np.cumsum(chunk_histograms, axis=0)

    max_chunk_p = 0.0
    detected_chunk = False

    for hist in prefix_histograms:
        _, _, p_val = _compute_chi2_vectorized(hist)
        if p_val > max_chunk_p:
            max_chunk_p = p_val
        if p_val > 0.95:
            detected_chunk = True
            break

    stego_detected = (global_p > 0.95) or detected_chunk

    return {
        "stego_probability": max(global_p, max_chunk_p),
        "detected": stego_detected,
        "chi2_stat": global_chi2,
        "dof": global_dof,
        "total_ac_coefficients": len(nonzero_ac)
    }

def analyze_image(image_path: str, method: str = "auto") -> dict:
    ext = os.path.splitext(image_path)[1].lower()
    
    if method == "auto":
        if ext in [".jpg", ".jpeg"]:
            method = "jpeg_dct"
        elif ext in [".png"]:
            method = "png_lsb"
        else:
            raise ValueError(f"Unsupported image format for analysis: {ext}")

    if method == "jpeg_dct":
        return analyze_jpeg_dct(image_path)
    elif method == "png_lsb":
        return analyze_png_lsb(image_path)
