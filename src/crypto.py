from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from argon2 import PasswordHasher
from argon2.low_level import hash_secret_raw, Type
from typing import Optional
import struct

def generate_key(password: str, salt: Optional[bytes] = None) -> tuple[bytes, bytes]:
    if salt is None:
        salt = get_random_bytes(16)
    else:
        salt = bytes(salt)
    
    key = hash_secret_raw(
        password.encode(),
        salt, 
        time_cost=3, 
        memory_cost=65536, 
        parallelism=4, 
        hash_len=32,
        type=Type.ID)
    return key, salt

def encrypt_aes_gcm(plaintext: bytes, key: bytes) -> tuple[bytes, bytes, bytes]:
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(plaintext)
    return ciphertext, cipher.nonce, tag

def decrypt_aes_gcm(ciphertext: bytes, nonce: bytes, tag: bytes, key: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext

def encrypt_with_pwd(plaintext: bytes, password: str) -> bytes:
    key, salt = generate_key(password)
    ciphertext, nonce, tag = encrypt_aes_gcm(plaintext, key)
    result = salt + nonce + struct.pack('>I', len(ciphertext)) + ciphertext + tag
    return result

def decrypt_with_pwd(encrypted_data: bytes, password: str) -> bytes:
    encrypted_data = bytes(encrypted_data)
    salt = encrypted_data[:16]
    nonce = encrypted_data[16:32]
    ciphertext_length = struct.unpack('>I', encrypted_data[32:36])[0]
    ciphertext = encrypted_data[36:36+ciphertext_length]
    tag = encrypted_data[36+ciphertext_length:36+ciphertext_length+16]
    
    key, _ = generate_key(password, salt)
    plaintext = decrypt_aes_gcm(ciphertext, nonce, tag, key)
    return plaintext