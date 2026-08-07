"""
Phase 3 Tests: Comprehensive Suite for Spatial LSB, JPEG DCT (jpegio), and End-to-End Workflows.
"""

import sys
import os
import pathlib
import tempfile
import numpy as np
from PIL import Image

from stego.encoder import embed_bits_spatially, encode_with_rs, embed_bits_in_jpeg_dct
from stego.decoder import extract_bits_spatially, decode_with_rs, extract_bits_from_jpeg_dct
from stego.steganography import hide_message, extract_message
from stego.img_utils import load_image_as_rgb, save_rgb_as_png
from stego.helper import bytes_to_bits, bits_to_bytes

# ============================================================================
# Helpers
# ============================================================================

def create_dummy_png(path: str, size: tuple = (100, 100)):
    """Creates a synthetic RGB PNG for testing."""
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, 'RGB')
    img.save(path, 'PNG')

def create_dummy_jpeg(path: str, size: tuple = (128, 128)):
    """Creates a synthetic baseline JPEG for testing."""
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr, 'RGB')
    img.save(path, 'JPEG', quality=90)


# ============================================================================
# 1. Unit Tests: Spatial LSB Engine (PNG)
# ============================================================================

def test_spatial_lsb_vectorized():
    print("\n[TEST] Spatial LSB bit manipulation...")
    
    # 3D array representing 10x10 RGB image (300 values)
    img_data = np.full((10, 10, 3), 200, dtype=np.uint8)
    test_bits = [1, 0, 1, 1, 0, 0, 1, 0] * 5  # 40 bits
    
    modified = embed_bits_spatially(img_data, test_bits)
    extracted = extract_bits_spatially(modified, len(test_bits))
    
    assert extracted == test_bits, "Extracted spatial bits do not match embedded bits"
    print("✓ Spatial LSB bit manipulation works losslessly")

def test_png_rgb_io_roundtrip():
    print("\n[TEST] PNG RGB I/O lossless roundtrip...")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_png = tmp.name
        
    try:
        create_dummy_png(tmp_png, (64, 64))
        original_data = load_image_as_rgb(tmp_png)
        
        test_bits = [1, 0, 1, 1, 0, 1, 0, 0] * 10
        modified_data = embed_bits_spatially(original_data, test_bits)
        save_rgb_as_png(modified_data, tmp_png)
        
        reloaded_data = load_image_as_rgb(tmp_png)
        extracted_bits = extract_bits_spatially(reloaded_data, len(test_bits))
        
        assert extracted_bits == test_bits, "PNG save/reload altered spatial LSB bits"
        print("✓ PNG RGB I/O preserves embedded LSB bits perfectly")
    finally:
        if os.path.exists(tmp_png):
            os.unlink(tmp_png)


# ============================================================================
# 2. Unit Tests: Quantized JPEG DCT Engine (jpegio)
# ============================================================================

def test_jpeg_dct_jpegio_roundtrip():
    print("\n[TEST] JPEG Quantized DCT (jpegio) roundtrip...")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        in_jpg = tmp_in.name
        out_jpg = tmp_out.name

    try:
        create_dummy_jpeg(in_jpg, (128, 128))
        
        test_bits = [1, 1, 0, 0, 1, 0, 1, 0] * 20  # 160 bits
        embed_bits_in_jpeg_dct(in_jpg, test_bits, out_jpg)
        
        extracted_bits = extract_bits_from_jpeg_dct(out_jpg, len(test_bits))
        
        assert extracted_bits == test_bits, "Extracted JPEG AC coefficients do not match embedded bits"
        print("✓ JPEG DCT coefficient embedding/extraction works losslessly")
    finally:
        for p in [in_jpg, out_jpg]:
            if os.path.exists(p):
                os.unlink(p)


# ============================================================================
# 3. Integration Tests: End-to-End Workflows (steganography.py)
# ============================================================================

def test_e2e_png_lsb_workflow():
    print("\n[TEST] End-to-End PNG LSB Hide & Extract...")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        input_png = tmp.name
        
    try:
        create_dummy_png(input_png, (100, 100))
        secret_msg = "Steganography testing for PNG LSB!"
        password = "SecurePassword123"
        
        res = hide_message(input_png, secret_msg, password, method="png_lsb")
        stego_path = res["output_path"]
        
        extracted_msg = extract_message(stego_path, password, method="png_lsb")
        
        assert extracted_msg == secret_msg, f"E2E PNG failed: '{extracted_msg}' != '{secret_msg}'"
        print("✓ E2E PNG LSB Hide & Extract passed")
    finally:
        if os.path.exists(input_png):
            os.unlink(input_png)
        if 'stego_path' in locals() and os.path.exists(stego_path):
            os.unlink(stego_path)

def test_e2e_jpeg_dct_workflow():
    print("\n[TEST] End-to-End JPEG DCT Hide & Extract...")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        input_jpg = tmp.name
        
    try:
        create_dummy_jpeg(input_jpg, (128, 128))
        secret_msg = "Steganography testing for JPEG DCT!"
        password = "SecurePassword456"
        
        res = hide_message(input_jpg, secret_msg, password, method="jpeg_dct")
        stego_path = res["output_path"]
        
        extracted_msg = extract_message(stego_path, password, method="jpeg_dct")
        
        assert extracted_msg == secret_msg, f"E2E JPEG failed: '{extracted_msg}' != '{secret_msg}'"
        print("✓ E2E JPEG DCT Hide & Extract passed")
    finally:
        if os.path.exists(input_jpg):
            os.unlink(input_jpg)
        if 'stego_path' in locals() and os.path.exists(stego_path):
            os.unlink(stego_path)

def test_e2e_auto_method_selection():
    print("\n[TEST] End-to-End Auto Method Selection...")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_jpg:
        png_path = tmp_png.name
        jpg_path = tmp_jpg.name

    try:
        create_dummy_png(png_path, (80, 80))
        create_dummy_jpeg(jpg_path, (128, 128))
        
        res_png = hide_message(png_path, "Auto PNG", "pwd", method="auto")
        res_jpg = hide_message(jpg_path, "Auto JPEG", "pwd", method="auto")
        
        assert res_png["method_used"] == "png_lsb", f"Expected png_lsb, got {res_png['method_used']}"
        assert res_jpg["method_used"] == "jpeg_dct", f"Expected jpeg_dct, got {res_jpg['method_used']}"
        
        msg_png = extract_message(res_png["output_path"], "pwd", method="auto")
        msg_jpg = extract_message(res_jpg["output_path"], "pwd", method="auto")
        
        assert msg_png == "Auto PNG"
        assert msg_jpg == "Auto JPEG"
        print("✓ Automatic method resolution works correctly")
    finally:
        for p in [png_path, jpg_path, res_png.get("output_path"), res_jpg.get("output_path")]:
            if p and os.path.exists(p):
                os.unlink(p)


# ============================================================================
# 4. Error Handling & Edge Case Tests
# ============================================================================

def test_invalid_header_error():
    print("\n[TEST] Invalid Header Detection...")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        clean_png = tmp.name
        
    try:
        create_dummy_png(clean_png, (50, 50))
        
        # Attempt to extract from an image with no hidden data
        try:
            extract_message(clean_png, "password123", method="png_lsb")
            assert False, "Should have raised ValueError for invalid header"
        except ValueError as e:
            assert "Invalid payload header" in str(e)
            print("✓ Invalid payload header correctly caught")
    finally:
        if os.path.exists(clean_png):
            os.unlink(clean_png)

def test_capacity_overflow():
    print("\n[TEST] Capacity Overflow Handling...")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tiny_png = tmp.name
        
    try:
        # 10x10 RGB PNG has 300 bits total capacity
        create_dummy_png(tiny_png, (10, 10))
        huge_message = "A" * 1000  # Will generate a payload larger than 300 bits
        
        try:
            hide_message(tiny_png, huge_message, "pwd", method="png_lsb")
            assert False, "Should have raised ValueError for capacity overflow"
        except ValueError as e:
            assert "Message is too large" in str(e) or "too large" in str(e)
            print("✓ Capacity overflow correctly caught")
    finally:
        if os.path.exists(tiny_png):
            os.unlink(tiny_png)


# ============================================================================
# Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 3 TEST SUITE: Spatial LSB, JPEG DCT, and Steganography Core")
    print("=" * 70)
    
    try:
        test_spatial_lsb_vectorized()
        test_png_rgb_io_roundtrip()
        test_jpeg_dct_jpegio_roundtrip()
        test_e2e_png_lsb_workflow()
        test_e2e_jpeg_dct_workflow()
        test_e2e_auto_method_selection()
        test_invalid_header_error()
        test_capacity_overflow()
        
        print("\n" + "=" * 70)
        print("✓ ALL PHASE 3 TESTS PASSED SUCCESSFULLY")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
