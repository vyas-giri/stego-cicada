import struct
import os
from datetime import datetime
from stego.crypto import decrypt_with_pwd, encrypt_with_pwd
from stego.encoder import encode_with_rs, embed_bits_spatially, embed_bits_in_jpeg_dct
from stego.helper import bytes_to_bits, bits_to_bytes
from stego.img_utils import save_rgb_as_png, load_image_as_rgb
from stego.decoder import decode_with_rs, extract_bits_spatially, extract_bits_from_jpeg_dct
from stego.constants import OUTPUT_DIR, MAGIC

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def resolve_method(image_path: str, method: str) -> str:
    if method == "auto":
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            return "jpeg_dct"
        elif ext in [".png"]:
            return "png_lsb"
        else:
            raise ValueError(f"Unsupported image format: {ext}")
    return method


def hide_message(
        image_path: str,
        message: str,
        password: str,
        method: str = "auto"
) -> dict:

    selected_method = resolve_method(image_path, method)
    
    # encrypt the message with password and encode with Reed-Solomon
    bytes_message = message.encode('utf-8')
    encrypted_message = encrypt_with_pwd(bytes_message, password)
    rs_encoded_message = encode_with_rs(encrypted_message)
    header = MAGIC + struct.pack('>I', len(rs_encoded_message))
    payload = header + rs_encoded_message
    bits_payload = bytes_to_bits(payload)

    if selected_method == "jpeg_dct":
        output_path = os.path.join(OUTPUT_DIR, f"stego_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        embed_bits_in_jpeg_dct(image_path, bits_payload, output_path)
        capacity_bits = os.path.getsize(image_path) * 8  # Approximate capacity in bits for JPEG

    elif selected_method == "png_lsb":
        output_path = os.path.join(OUTPUT_DIR, f"stego_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        image_rgb = load_image_as_rgb(image_path)
        capacity_bits = image_rgb.size  # Total number of bits available in the image

        if len(bits_payload) > capacity_bits:
            raise ValueError(f"Message is too large. Required: {len(bits_payload)} bits, Available: {capacity_bits} bits.")

        modified_rgb = embed_bits_spatially(image_rgb, bits_payload)
        save_rgb_as_png(modified_rgb, output_path)

    else:
        raise ValueError(f"Unsupported method: {selected_method}")

    return {
        "output_path": output_path,
        "method_used": selected_method,
        "original_image_size": os.path.getsize(image_path),
        "modified_image_size": os.path.getsize(output_path),
        "message_length": len(message),
        "encrypted_length": len(encrypted_message),
        "rs_encoded_length": len(rs_encoded_message),
        "bits_length": len(bits_payload),
        "capacity_used": (len(bits_payload) / capacity_bits) * 100,
    }

def extract_message(
        image_path: str,
        password: str,
        method: str = "auto",
) -> str:
    selected_method = resolve_method(image_path, method)

    if selected_method == "png_lsb":
        image_rgb = load_image_as_rgb(image_path)
        header_bits = extract_bits_spatially(image_rgb, 64)  # Extract first 64 bits for header
    elif selected_method == "jpeg_dct":
        header_bits = extract_bits_from_jpeg_dct(image_path, 64)
    else:
        raise ValueError(f"Unsupported method: {selected_method}")
    
    header_bytes = bits_to_bytes(header_bits)
    if header_bytes[:4] != MAGIC:
        raise ValueError("Invalid payload header.")

    payload_length = struct.unpack('>I', header_bytes[4:8])[0]
    total_bits = (8 + payload_length) * 8

    if selected_method == "png_lsb":
        extracted_bits = extract_bits_spatially(image_rgb, total_bits)
    elif selected_method == "jpeg_dct":
        extracted_bits = extract_bits_from_jpeg_dct(image_path, total_bits)
    
    extracted_bytes = bits_to_bytes(extracted_bits)

    rs_decoded_bytes = decode_with_rs(extracted_bytes[8:8 + payload_length])
    decrypted_message = decrypt_with_pwd(rs_decoded_bytes, password)
    return decrypted_message.decode('utf-8', errors='ignore')