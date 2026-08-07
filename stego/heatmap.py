import os
import numpy as np
from PIL import Image

def generate_residual_heatmap(
    original_path: str,
    stego_path: str,
    output_path: str,
    amplification: int = 50
) -> str:
    """
    Calculates pixel-wise residuals between original and stego images.
    Highlights modified pixels in high-contrast bright red and logs bounding box dimensions.
    """
    img_org = Image.open(original_path).convert('RGB')
    img_stego = Image.open(stego_path).convert('RGB')

    arr_org = np.array(img_org, dtype=np.int16)
    arr_stego = np.array(img_stego, dtype=np.int16)

    if arr_org.shape != arr_stego.shape:
        raise ValueError("Image dimensions do not match for residual comparison.")

    # Calculate absolute differences
    diff = np.abs(arr_org - arr_stego)
    
    # Mask pixels where any RGB channel was modified
    modified_mask = np.any(diff > 0, axis=2)
    coords = np.argwhere(modified_mask)

    if len(coords) == 0:
        print("[HEATMAP INFO] No pixel differences detected between original and stego image.")
        heatmap = np.zeros_like(arr_org, dtype=np.uint8)
    else:
        ymin, xmin = coords.min(axis=0)
        ymax, xmax = coords.max(axis=0)
        
        total_pixels = arr_org.shape[0] * arr_org.shape[1]
        print(f"[HEATMAP INFO] Total modified pixels: {len(coords)} / {total_pixels} ({len(coords)/total_pixels*100:.4f}% of image)")
        print(f"[HEATMAP INFO] Bounding Box: X=[{xmin}, {xmax}], Y=[{ymin}, {ymax}]")

        # Create bright red highlights against black background
        heatmap = np.zeros_like(arr_org, dtype=np.uint8)
        heatmap[modified_mask] = [255, 0, 0]  # Bright Red

        # Optional: Expand thin lines with a 3x3 dilation for visual visibility on 4K/HD displays
        padded_mask = np.pad(modified_mask, 1, mode='constant')
        dilated_mask = (
            padded_mask[:-2, :-2] | padded_mask[:-2, 1:-1] | padded_mask[:-2, 2:] |
            padded_mask[1:-1, :-2] | padded_mask[1:-1, 1:-1] | padded_mask[1:-1, 2:] |
            padded_mask[2:, :-2]  | padded_mask[2:, 1:-1]  | padded_mask[2:, 2:]
        )
        heatmap[dilated_mask] = [255, 0, 0]

    diff_image = Image.fromarray(heatmap, 'RGB')
    diff_image.save(output_path)

    return output_path