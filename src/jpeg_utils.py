from PIL import Image
import numpy as np

def load_jpeg_as_ycbcr(filepath: str) -> np.ndarray:
    img = Image.open(filepath)
    if img.mode != 'YCbCr':
        img = img.convert('YCbCr')
    return np.array(img)

def save_ycbcr_as_jpeg(data: np.ndarray, filepath: str, quality: int = 95):
    data = np.clip(data, 0, 255).astype(np.uint8)
    img = Image.fromarray(data, 'YCbCr')
    img.save(filepath, 'JPEG', quality=quality)

def inspect_jpeg(filepath: str):
    img = Image.open(filepath)
    print(f"Format: {img.format}")
    print(f"Mode: {img.mode}")
    print(f"Size: {img.size}")
    print(f"Info: {img.info}")

def split_into_blocks(y_channel: np.ndarray, block_size: int = 8) -> np.ndarray:
    h, w = y_channel.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    blocks = y_channel[:h_blocks*block_size, :w_blocks*block_size].reshape(h_blocks, block_size, w_blocks, block_size)
    return blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)

def merge_blocks(blocks: np.ndarray, w_blocks: int, block_size: int = 8) -> np.ndarray:
    total_blocks = blocks.shape[0]
    h_blocks = total_blocks // w_blocks
    merged = blocks.reshape(h_blocks, w_blocks, block_size, block_size).transpose(0, 2, 1, 3)
    cropped_h = h_blocks * block_size
    cropped_w = w_blocks * block_size
    return merged.reshape(cropped_h, cropped_w)
