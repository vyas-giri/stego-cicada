import numpy as np
from reedsolo import RSCodec
from typing import cast

def decode_with_rs(encoded_data: bytes, ecc: int = 32) -> bytes:
    rsc = RSCodec(ecc)
    decoded_data = rsc.decode(encoded_data)[0]
    return cast(bytes, decoded_data)

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

def extract_bits_spatially(y_channel: np.ndarray, num_bits: int) -> list:
    flat_y = y_channel.flatten().astype(np.uint8)

    if num_bits > len(flat_y):
        raise ValueError("Requested more bits than available pixels.")

    extracted_bits = (flat_y[:num_bits] & 1).tolist()
    return extracted_bits
