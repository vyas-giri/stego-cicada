import numpy as np
from scipy.fft import dctn, idctn
from typing import cast

# Re-export embedding helpers so tests can import them from `src.dct`
from helper import get_embedding_mask as _helper_get_embedding_mask
from encoder import embed_bits_in_blocks as _embed_bits_in_blocks
from decoder import extract_bits_from_blocks as _extract_bits_from_blocks

def apply_dct_to_blocks(blocks: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, dctn(blocks, axes=(1, 2), norm='ortho'))

def apply_idct_to_blocks(dct_blocks: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, idctn(dct_blocks, axes=(1, 2), norm='ortho'))

def get_embedding_mask() -> np.ndarray:
    return _helper_get_embedding_mask()


def embed_bits_in_blocks(blocks: np.ndarray, bits: list, mask: np.ndarray, strength: float = 5.0) -> np.ndarray:
    return _embed_bits_in_blocks(blocks, bits, mask, strength=strength)


def extract_bits_from_blocks(blocks: np.ndarray, num_bits: int, mask: np.ndarray) -> list:
    return _extract_bits_from_blocks(blocks, num_bits, mask)
