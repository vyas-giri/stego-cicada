import numpy as np
import jpegio
from scipy.stats import chi2
from PIL import Image
import os

def analyze_jpeg_dct(image_path: str) -> dict:
    """
    Performs Chi-Square steganalysis on quantized JPEG AC DCT coefficients.
    Returns chi2 statistic, degrees of freedom, and stego probability p.
    """

    jpeg_obj = jpegio.read(image_path)
    y_coefs = jpeg_obj.coef_arrays[0] # 2D int16 matrix

    h, w = y_coefs.shape

    ac_mask = np.ones((h, w), dtype=bool)
    ac_mask[::8, ::8] = False

    ac_coefs = y_coefs[ac_mask]

    nonzero_ac = ac_coefs[ac_coefs != 0]

    uniqueVals, counts = np.unique(nonzero_ac, return_counts=True)
    vapMap = dict(zip(uniqueVals, counts))

    chi2_stat = 0.0
    valid_pairs = 0

    all_bases = np.unique(nonzero_ac & ~1)

    for base in all_bases:
        val_even = base
        val_odd = base | 1

        n_even = vapMap.get(val_even, 0)
        n_odd = vapMap.get(val_odd, 0)
        totalPairCnt = n_even + n_odd

        if totalPairCnt >= 10:
            chi2_stat += ((n_even - n_odd) ** 2) / totalPairCnt
            valid_pairs += 1

    if valid_pairs == 0:
        return {"p_value": 0.0, "detected": False, "chi2_stat": 0.0, "dof": 0}

    p_value = float(chi2.sf(chi2_stat, df=valid_pairs))

    return {
        "stego_probability": p_value,
        "detected": p_value > 0.95,
        "chi2_stat": float(chi2_stat),
        "dof": valid_pairs,
        "total_ac_coefficients": len(ac_coefs)
    }

def analyze_png_lsb(image_path: str) -> dict:
    """
    Performs Chi-Square steganalysis on Spatial RGB PNG images.
    """
    img = Image.open(image_path).convert('RGB')
    flat_pixels = np.array(img).flatten()

    unique_vals, counts = np.unique(flat_pixels, return_counts=True)
    val_map = dict(zip(unique_vals, counts))

    chi2_stat = 0.0
    valid_pairs = 0

    for base in range(0, 256, 2):
        n_even = val_map.get(base, 0)
        n_odd = val_map.get(base + 1, 0)
        total_pair_count = n_even + n_odd

        if total_pair_count >= 10:
            chi2_stat += ((n_even - n_odd) ** 2) / total_pair_count
            valid_pairs += 1

    if valid_pairs == 0:
        return {"stego_probability": 0.0, "detected": False, "chi2_stat": 0.0, "dof": 0}

    p_value = float(chi2.sf(chi2_stat, valid_pairs))

    return {
        "stego_probability": p_value,
        "detected": p_value > 0.95,
        "chi2_stat": float(chi2_stat),
        "dof": valid_pairs
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
