"""
Phase 2 Tests: DCT Embedding & Extraction
Tests block splitting, DCT transforms, bit embedding, and JPEG roundtrip.
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from src.dct import (
    apply_dct_to_blocks, apply_idct_to_blocks, 
    get_embedding_mask, embed_bits_in_blocks, extract_bits_from_blocks
)
from src.jpeg_utils import (
    split_into_blocks, merge_blocks, 
    load_jpeg_as_ycbcr, save_ycbcr_as_jpeg, inspect_jpeg
)
import tempfile
import os


# ============================================================================
# Helper Functions
# ============================================================================

def bytes_to_bits(data: bytes) -> list:
    """Convert bytes to list of bits (0 or 1)."""
    return [int(b) for byte in data for b in bin(byte)[2:].zfill(8)]


def bits_to_bytes(bits: list) -> bytes:
    """Convert list of bits back to bytes."""
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - len(bits) % 8)  # Pad with zeros
    return bytes(int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (higher is better, >= 40 dB is good)."""
    original = original.astype(float)
    modified = modified.astype(float)
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100.0  # Identical images
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (higher is better, >= 0.95 is good).
    Simplified version without scipy dependency.
    """
    original = original.astype(float)
    modified = modified.astype(float)
    
    c1, c2 = 6.5025, 58.5225  # Constants
    mean_orig = np.mean(original)
    mean_mod = np.mean(modified)
    var_orig = np.var(original)
    var_mod = np.var(modified)
    cov = np.mean((original - mean_orig) * (modified - mean_mod))
    
    ssim = ((2 * mean_orig * mean_mod + c1) * (2 * cov + c2)) / \
           ((mean_orig**2 + mean_mod**2 + c1) * (var_orig + var_mod + c2))
    return float(np.clip(ssim, 0, 1))


# ============================================================================
# Tests: Block Operations
# ============================================================================

def test_split_merge_blocks_consistency():
    """Test that split + merge returns original data."""
    print("\n[TEST] Block split/merge consistency...")
    
    # Create synthetic Y channel (1080 x 1920)
    y_channel = np.random.randint(0, 256, (1080, 1920), dtype=np.uint8)
    
    # Split into blocks
    h_blocks = 1080 // 8
    w_blocks = 1920 // 8
    blocks = split_into_blocks(y_channel)
    
    # Merge back
    y_reconstructed = merge_blocks(blocks, w_blocks, block_size=8)
    
    # Check shape and values
    assert blocks.shape == (h_blocks * w_blocks, 8, 8), f"Expected ({h_blocks * w_blocks}, 8, 8), got {blocks.shape}"
    assert y_reconstructed.shape == (1080, 1920), f"Shape mismatch: {y_reconstructed.shape}"
    
    # Check values match (should be identical for uint8)
    assert np.allclose(y_channel, y_reconstructed), "Reconstructed blocks don't match original"
    
    print("✓ Block split/merge preserves data perfectly")


# ============================================================================
# Tests: DCT Transform
# ============================================================================

def test_dct_idct_roundtrip():
    """Test that DCT + IDCT is lossless (within floating point precision)."""
    print("\n[TEST] DCT/IDCT roundtrip (no embedding)...")
    
    # Create synthetic blocks
    blocks = np.random.randn(1000, 8, 8).astype(np.float32)  # 1000 8x8 blocks
    
    # DCT
    dct_blocks = apply_dct_to_blocks(blocks)
    
    # IDCT
    reconstructed = apply_idct_to_blocks(dct_blocks)
    
    # Check close to original (ortho norm ensures this)
    mse = np.mean((blocks - reconstructed) ** 2)
    print(f"  MSE (DCT roundtrip): {mse:.2e}")
    assert mse < 1e-10, f"DCT roundtrip error too large: {mse}"
    
    print("✓ DCT/IDCT is lossless within floating point precision")


# ============================================================================
# Tests: Embedding & Extraction (No JPEG)
# ============================================================================

def test_embed_extract_bits():
    """Test embedding and extracting bits without JPEG quantization."""
    print("\n[TEST] Embed/extract bits (raw DCT, no JPEG)...")
    
    # Create synthetic DCT blocks
    blocks = np.random.randn(1000, 8, 8) * 50  # Realistic DCT coefficient range
    
    # Get mask
    mask = get_embedding_mask()
    capacity_per_block = np.sum(mask)
    total_capacity = capacity_per_block * blocks.shape[0]
    
    print(f"  Mask capacity per block: {capacity_per_block} bits")
    print(f"  Total capacity: {total_capacity} bits")
    
    # Generate random bits to embed
    num_bits_to_embed = min(1000, total_capacity)
    test_bits = [np.random.randint(0, 2) for _ in range(num_bits_to_embed)]
    
    # Embed
    modified_blocks = embed_bits_in_blocks(blocks, test_bits, mask)
    
    # Extract
    extracted_bits = extract_bits_from_blocks(modified_blocks, num_bits_to_embed, mask)
    
    # Verify
    accuracy = sum(e == t for e, t in zip(extracted_bits, test_bits)) / len(test_bits)
    print(f"  Extraction accuracy: {accuracy * 100:.1f}%")
    assert accuracy == 1.0, f"Bit extraction failed: {accuracy * 100:.1f}% accuracy"
    
    print("✓ Embed/extract bits works perfectly (no JPEG)")


def test_embed_extract_bytes():
    """Test embedding and extracting actual encrypted bytes."""
    print("\n[TEST] Embed/extract encrypted bytes (raw DCT, no JPEG)...")
    
    # Simulate encrypted data
    test_message = b"This is a secret message for steganography!"
    test_bits = bytes_to_bits(test_message)
    
    # Create DCT blocks
    blocks = np.random.randn(2000, 8, 8) * 50
    mask = get_embedding_mask()
    
    # Embed
    modified_blocks = embed_bits_in_blocks(blocks, test_bits, mask)
    
    # Extract
    extracted_bits = extract_bits_from_blocks(modified_blocks, len(test_bits), mask)
    extracted_bytes = bits_to_bytes(extracted_bits)
    
    # Verify
    assert extracted_bytes == test_message, f"Message mismatch: {extracted_bytes} != {test_message}"
    
    print(f"✓ Embedded and extracted: {test_message.decode()}")


# ============================================================================
# Tests: Full JPEG Roundtrip
# ============================================================================

def test_full_jpeg_roundtrip():
    """Test the complete pipeline: load image → embed → save JPEG → load → extract."""
    print("\n[TEST] Full JPEG roundtrip (embed → save → reload → extract)...")
    
    # Load real JPEG
    ycbcr = load_jpeg_as_ycbcr("data/input_imgs/chilling.jpg")
    original_ycbcr = ycbcr.copy()
    y_original = ycbcr[:, :, 0]
    
    print(f"  Loaded image shape: {ycbcr.shape}, Y range: [{y_original.min()}, {y_original.max()}]")
    
    # Phase 2: Embed
    h_blocks = y_original.shape[0] // 8
    w_blocks = y_original.shape[1] // 8
    blocks = split_into_blocks(y_original)
    dct_blocks = apply_dct_to_blocks(blocks)
    
    # Create test payload
    test_message = b"Secret data embedded via DCT!"
    test_bits = bytes_to_bits(test_message)
    
    mask = get_embedding_mask()
    capacity = np.sum(mask) * blocks.shape[0]
    print(f"  Capacity: {capacity} bits, message: {len(test_bits)} bits")
    
    assert len(test_bits) <= capacity, "Message too large for capacity"
    
    # Embed
    dct_modified = embed_bits_in_blocks(dct_blocks, test_bits, mask)
    idct_blocks = apply_idct_to_blocks(dct_modified)
    y_modified = merge_blocks(idct_blocks, w_blocks, block_size=8)
    ycbcr[:, :, 0] = y_modified.astype(np.uint8)
    
    # Compute quality metrics before save
    psnr_before = compute_psnr(y_original, y_modified)
    ssim_before = compute_ssim(y_original, y_modified)
    print(f"  Before JPEG save - PSNR: {psnr_before:.2f} dB, SSIM: {ssim_before:.4f}")
    
    # Save as JPEG with different quality levels and test extraction
    for quality in [95, 85, 75]:
        print(f"\n  Testing JPEG quality={quality}...")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save
            save_ycbcr_as_jpeg(ycbcr, tmp_path, quality=quality)
            
            # Reload
            ycbcr_reloaded = load_jpeg_as_ycbcr(tmp_path)
            y_reloaded = ycbcr_reloaded[:, :, 0]
            
            # Compute quality metrics after reload
            psnr_after = compute_psnr(y_original, y_reloaded)
            ssim_after = compute_ssim(y_original, y_reloaded)
            print(f"    After JPEG reload - PSNR: {psnr_after:.2f} dB, SSIM: {ssim_after:.4f}")
            
            # Extract
            blocks_reloaded = split_into_blocks(y_reloaded)
            dct_reloaded = apply_dct_to_blocks(blocks_reloaded)
            extracted_bits = extract_bits_from_blocks(dct_reloaded, len(test_bits), mask)
            extracted_bytes = bits_to_bytes(extracted_bits)
            
            # Verify
            accuracy = sum(e == t for e, t in zip(extracted_bits, test_bits)) / len(test_bits)
            print(f"    Extraction accuracy: {accuracy * 100:.1f}%")
            
            if accuracy == 1.0:
                print(f"    ✓ Message recovered: {extracted_bytes.decode()}")
            else:
                print(f"    ✗ Message corrupted (accuracy: {accuracy * 100:.1f}%)")
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_capacity_and_quality_tradeoff():
    """Test how embedding strength affects image quality and robustness."""
    print("\n[TEST] Capacity/quality tradeoff analysis...")
    
    # Load image
    ycbcr = load_jpeg_as_ycbcr("data/input_imgs/chilling.jpg")
    y_original = ycbcr[:, :, 0].copy()
    
    # Embed with current mask
    h_blocks = y_original.shape[0] // 8
    w_blocks = y_original.shape[1] // 8
    blocks = split_into_blocks(y_original)
    dct_blocks = apply_dct_to_blocks(blocks)
    
    mask = get_embedding_mask()
    capacity = np.sum(mask) * blocks.shape[0]
    
    print(f"  Capacity per block: {np.sum(mask)} bits")
    print(f"  Total blocks: {blocks.shape[0]}")
    print(f"  Total capacity: {capacity} bits ({capacity // 8} bytes)")
    
    # Generate test message
    test_message = b"A" * (capacity // 8 - 10)  # Leave some margin
    test_bits = bytes_to_bits(test_message)
    
    # Embed
    dct_modified = embed_bits_in_blocks(dct_blocks, test_bits, mask)
    idct_blocks = apply_idct_to_blocks(dct_modified)
    y_modified = merge_blocks(idct_blocks, w_blocks, block_size=8).astype(np.uint8)
    
    # Quality metrics
    psnr = compute_psnr(y_original, y_modified)
    ssim = compute_ssim(y_original, y_modified)
    
    print(f"  Image quality after embedding:")
    print(f"    PSNR: {psnr:.2f} dB (target: > 40 dB)")
    print(f"    SSIM: {ssim:.4f} (target: > 0.95)")
    
    assert psnr > 40, f"PSNR too low: {psnr:.2f}"
    assert ssim > 0.95, f"SSIM too low: {ssim:.4f}"
    print("  ✓ Quality metrics acceptable")


# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 2 TEST SUITE: DCT Embedding & JPEG Steganography")
    print("=" * 70)
    
    try:
        # Block operations
        test_split_merge_blocks_consistency()
        
        # DCT transforms
        test_dct_idct_roundtrip()
        
        # Embedding/extraction (raw)
        test_embed_extract_bits()
        test_embed_extract_bytes()
        
        # Full pipeline
        test_full_jpeg_roundtrip()
        test_capacity_and_quality_tradeoff()
        
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
