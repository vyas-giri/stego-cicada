from src.crypto import encrypt_with_pwd as encrypt_with_password, decrypt_with_pwd as decrypt_with_password

def test_encrypt_decrypt():
    message = b"Hello, secret world!"
    password = "mypassword123"
    
    # Encrypt
    encrypted = encrypt_with_password(message, password)
    print(f"Original: {message}")
    print(f"Encrypted (hex): {encrypted.hex()[:50]}...")
    
    # Decrypt
    decrypted = decrypt_with_password(encrypted, password)
    print(f"Decrypted: {decrypted}")
    
    assert decrypted == message
    print("✓ Encryption/decryption works!")

def test_wrong_password():
    message = b"Secret"
    encrypted = encrypt_with_password(message, "correctpassword")
    
    try:
        decrypt_with_password(encrypted, "wrongpassword")
        assert False, "Should have raised an error"
    except ValueError:
        print("✓ Wrong password correctly rejected!")

if __name__ == "__main__":
    test_encrypt_decrypt()
    test_wrong_password()