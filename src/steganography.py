import struct
import os
from datetime import datetime
import numpy as np

from crypto import decrypt_with_pwd, encrypt_with_pwd
from encoder import encode_with_rs
from helper import bytes_to_bits, get_embedding_mask, bits_to_bytes
from img_utils import save_rgb_as_png, load_image_as_rgb
from encoder import embed_bits_in_blocks, embed_bits_spatially
from dct import apply_dct_to_blocks, apply_idct_to_blocks
from decoder import decode_with_rs, extract_bits_from_blocks, extract_bits_spatially
from constants import OUTPUT_DIR, MAGIC

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def hide_message(
        image_path: str,
        message: str,
        password: str,
) -> dict:

    OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"stego_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    
    # encrypt the message with password and encode with Reed-Solomon
    bytes_message = message.encode('utf-8')
    encrypted_message = encrypt_with_pwd(bytes_message, password)
    rs_encoded_message = encode_with_rs(encrypted_message)
    header = MAGIC + struct.pack('>I', len(rs_encoded_message))
    print(f"DEBUG: header bytes {header.hex()}")
    payload = header + rs_encoded_message
    bits_payload = bytes_to_bits(payload)

    # Check if the image has enough capacity to hold the message
    required_capacity = len(bits_payload)

    image_rgb = load_image_as_rgb(image_path)

    capacity_bits = image_rgb.size
    if required_capacity > capacity_bits:
        raise ValueError(f"Message is too large. Required: {required_capacity} bits, Available: {capacity_bits} bits.")

    modified_rgb = embed_bits_spatially(image_rgb, bits_payload)
    save_rgb_as_png(modified_rgb, OUTPUT_PATH)

    return {
        "output_path": OUTPUT_PATH,
        "original_image_size": os.path.getsize(image_path),
        "modified_image_size": os.path.getsize(OUTPUT_PATH),
        "message_length": len(message),
        "encrypted_length": len(encrypted_message),
        "rs_encoded_length": len(rs_encoded_message),
        "bits_length": len(bits_payload),
        "capacity_used": (len(bits_payload) / capacity_bits) * 100,
    }

def extract_message(
        image_path: str,
        password: str,
        num_bits: int | None = None
) -> str:
    image_rgb = load_image_as_rgb(image_path)

    header_bits = extract_bits_spatially(image_rgb, 64)  # Extract first 64 bits for header
    header_bytes = bits_to_bytes(header_bits)
    header_hex = header_bytes[:8].hex()

    print(f"DEBUG: extracted header bytes {header_hex}")
    if header_bytes[:4] != MAGIC:
        raise ValueError("Invalid payload header.")

    payload_length = struct.unpack('>I', header_bytes[4:8])[0]
    total_bits = (8 + payload_length) * 8

    if num_bits is not None:
        total_bits = max(total_bits, num_bits)

    extracted_bits = extract_bits_spatially(image_rgb, total_bits)
    extracted_bytes = bits_to_bytes(extracted_bits)

    if len(extracted_bytes) < 8 + payload_length:
        raise ValueError("Extracted payload length exceeds available data.")

    extracted_bytes = extracted_bytes[8:8 + payload_length]
    rs_decoded_bytes = decode_with_rs(extracted_bytes)

    decrypted_message = decrypt_with_pwd(rs_decoded_bytes, password)
    decoded_message = decrypted_message.decode('utf-8', errors='ignore')
    return decoded_message