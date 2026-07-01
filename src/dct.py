import numpy as np
from scipy.fft import dctn, idctn
from typing import cast

def apply_dct_to_blocks(blocks: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, dctn(blocks, axes=(1, 2), norm='ortho'))

def apply_idct_to_blocks(dct_blocks: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, idctn(dct_blocks, axes=(1, 2), norm='ortho'))

def get_embedding_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2, 3] = 1
    mask[3, 3] = 1
    mask[3, 4] = 1
    mask[4, 3] = 1
    mask[4, 4] = 1
    return mask

# LSB embedding in DCT coefficients
def embed_bits_in_blocks(blocks: np.ndarray, bits: list, mask: np.ndarray, strength: float = 5.0) -> np.ndarray:
    modified_blocks = blocks.copy()
    bit_index = 0
    for block_idx in range(modified_blocks.shape[0]):
        for i in range(8):
            for j in range(8):
                if mask[i, j] == 1 and bit_index < len(bits):
                    coeff = int(np.round(modified_blocks[block_idx, i, j]))
                    if bits[bit_index] == 1:
                        coeff |= 1  # Set LSB to 1
                    else:
                        coeff &= ~1  # Set LSB to 0
                    modified_blocks[block_idx, i, j] = float(coeff)
                    bit_index += 1
    return modified_blocks

def extract_bits_from_blocks(blocks: np.ndarray, num_bits: int, mask: np.ndarray) -> list:
    extracted_bits = []
    bit_index = 0
    for block_idx in range(blocks.shape[0]):
        for i in range(8):
            for j in range(8):
                if mask[i, j] == 1 and bit_index < num_bits:
                    coeff = int(np.round(blocks[block_idx, i, j]))
                    extracted_bits.append(coeff & 1)  # Extract LSB
                    bit_index += 1
    return extracted_bits
