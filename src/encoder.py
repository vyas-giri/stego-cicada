from reedsolo import RSCodec
from typing import cast
import numpy as np

def encode_with_rs(data: bytes, ecc: int = 32) -> bytes:
    rsc = RSCodec(ecc)
    encoded_data = rsc.encode(data)
    return cast(bytes, encoded_data)

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

def embed_bits_spatially(y_channel: np.ndarray, bits: list) -> np.ndarray:
    flat_y = y_channel.flatten().astype(np.uint8)

    if len(bits) > len(flat_y):
        raise ValueError(f"Payload too large. Required: {len(bits)} bits, Available: {len(flat_y)} bits.")
    
    flat_y[:len(bits)] &= 254
    flat_y[:len(bits)] |= np.array(bits, dtype=np.uint8)
    return flat_y.reshape(y_channel.shape)