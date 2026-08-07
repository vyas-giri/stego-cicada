import numpy as np
import jpegio
from scipy.stats import chi2
from PIL import Image
import os

def _compute_chi2_for_array(arr: np.ndarray) -> tuple[float, int, float]:
    """Computes Chi-Square statistic, degrees of freedom, and p-value safely across uint8 and int16."""
    if len(arr) == 0:
        return 0.0, 0, 0.0

    unique_vals, counts = np.unique(arr, return_counts=True)
    val_map = dict(zip(unique_vals, counts))

    chi2_stat = 0.0
    valid_pairs = 0

    # Type-safe LSB clearing: use 254 (0xFE) for uint8, ~1 for signed int16
    if arr.dtype == np.uint8:
        all_bases = np.unique(arr & 254)
    else:
        all_bases = np.unique(arr & ~1)

    for base in all_bases:
        val_even = base
        val_odd = base | 1

        n_even = val_map.get(val_even, 0)
        n_odd = val_map.get(val_odd, 0)
        total = n_even + n_odd

        if total >= 10:
            chi2_stat += ((n_even - n_odd) ** 2) / total
            valid_pairs += 1

    if valid_pairs == 0:
        return 0.0, 0, 0.0

    p_val = float(chi2.sf(chi2_stat, valid_pairs))
    return float(chi2_stat), valid_pairs, p_val

def analyze_jpeg_dct(image_path: str, chunk_size: int = 2048) -> dict:
    """Performs progressive and global Chi-Square steganalysis on non-zero JPEG AC coefficients."""
    jpeg_obj = jpegio.read(image_path)
    y_coefs = jpeg_obj.coef_arrays[0]

    h, w = y_coefs.shape
    ac_mask = np.ones((h, w), dtype=bool)
    ac_mask[::8, ::8] = False  # Exclude DC

    # Strictly evaluate non-zero AC coefficients
    nonzero_ac = y_coefs[ac_mask & (y_coefs != 0)]

    global_chi2, global_dof, global_p = _compute_chi2_for_array(nonzero_ac)

    # Scan sequentially in chunks to detect partial payloads
    max_chunk_p = 0.0
    detected_chunk = False

    for end_idx in range(chunk_size, len(nonzero_ac) + chunk_size, chunk_size):
        chunk = nonzero_ac[:min(end_idx, len(nonzero_ac))]
        _, _, p_val = _compute_chi2_for_array(chunk)
        if p_val > max_chunk_p:
            max_chunk_p = p_val
        if p_val > 0.95:
            detected_chunk = True
            break

    # An image is stego if pairs are equalized (p > 0.95) OR if the chi2 stat is massively distorted
    stego_detected = (global_p > 0.95) or detected_chunk or (global_chi2 > global_dof * 3)

    return {
        "stego_probability": max(global_p, max_chunk_p),
        "detected": stego_detected,
        "chi2_stat": global_chi2,
        "dof": global_dof,
        "total_ac_coefficients": len(nonzero_ac)
    }

def analyze_png_lsb(image_path: str, chunk_size: int = 4096) -> dict:
    """Performs progressive and global Chi-Square steganalysis on Spatial RGB PNG images."""
    img = Image.open(image_path).convert('RGB')
    flat_pixels = np.array(img).flatten()

    global_chi2, global_dof, global_p = _compute_chi2_for_array(flat_pixels)

    max_chunk_p = 0.0
    detected_chunk = False

    for end_idx in range(chunk_size, len(flat_pixels) + chunk_size, chunk_size):
        chunk = flat_pixels[:min(end_idx, len(flat_pixels))]
        _, _, p_val = _compute_chi2_for_array(chunk)
        if p_val > max_chunk_p:
            max_chunk_p = p_val
        if p_val > 0.95:
            detected_chunk = True
            break

    stego_detected = (global_p > 0.95) or detected_chunk or (global_chi2 > global_dof * 3)

    return {
        "stego_probability": max(global_p, max_chunk_p),
        "detected": stego_detected,
        "chi2_stat": global_chi2,
        "dof": global_dof
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
