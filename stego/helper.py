import numpy as np

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def bits_to_bytes(bits: list) -> bytes:
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte |= (bits[i + j] << (7 - j))
        bytes_list.append(byte)
    return bytes(bytes_list)

def get_embedding_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2, 3] = 1
    mask[3, 3] = 1
    mask[3, 4] = 1
    mask[4, 3] = 1
    mask[4, 4] = 1
    return mask
