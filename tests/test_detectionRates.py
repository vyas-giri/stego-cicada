import sys
import pathlib
import os
import tempfile
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from src.jpeg_engine import embed_bits_in_jpeg_dct
from src.steganalysis import analyze_jpeg_dct
from PIL import Image

def generate_test_jpeg(path: str, size=(256, 256)):
    # Create natural gradient JPEG to give realistic AC coefficient distributions
    x = np.linspace(0, 255, size[0], dtype=np.uint8)
    y = np.linspace(0, 255, size[1], dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr, 'RGB')
    img.save(path, 'JPEG', quality=85)

def run_detection_benchmark():
    print("=" * 70)
    print("JPEG DCT STEGANALYSIS DETECTION RATE BENCHMARK")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        in_jpg = tmp_in.name
        out_jpg = tmp_out.name

    try:
        generate_test_jpeg(in_jpg, (512, 512))

        # Check clean image baseline
        clean_res = analyze_jpeg_dct(in_jpg)
        print(f"Clean JPEG Baseline -> Stego Prob: {clean_res['stego_probability']:.4f} | Detected: {clean_res['detected']}")

        # Total capacity in AC coefficients
        import jpegio
        j = jpegio.read(in_jpg)
        ac_mask = np.ones(j.coef_arrays[0].shape, dtype=bool)
        ac_mask[::8, ::8] = False
        max_ac_bits = int(np.sum(ac_mask))

        payload_ratios = [0.05, 0.20, 0.50, 0.75, 1.00]

        print("\nTesting Payload Rates:")
        print(f"{'Payload %':<12} | {'Bits Embedded':<15} | {'p-value':<10} | {'Detected?'}")
        print("-" * 55)

        for ratio in payload_ratios:
            num_bits = int(max_ac_bits * ratio)
            bits = np.random.randint(0, 2, num_bits).tolist()

            embed_bits_in_jpeg_dct(in_jpg, bits, out_jpg)
            stego_res = analyze_jpeg_dct(out_jpg)

            prob = stego_res['stego_probability']
            detected = stego_res['detected']

            print(f"{ratio*100:>8.0f}%    | {num_bits:<15} | {prob:<10.4f} | {detected}")

    finally:
        for p in [in_jpg, out_jpg]:
            if os.path.exists(p):
                os.unlink(p)

if __name__ == "__main__":
    run_detection_benchmark()
