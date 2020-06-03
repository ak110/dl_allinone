def test_run():
    from Crypto import Random
    from Crypto.Cipher import AES
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Util.Padding import pad, unpad

    password = b"password1"
    plaintext = b"Hello World!"

    # encrypt
    iv = Random.get_random_bytes(AES.block_size)
    key = PBKDF2(password, salt=b"salt", dkLen=32)
    ciphertext = iv + AES.new(key, AES.MODE_CBC, iv=iv).encrypt(
        pad(plaintext, AES.block_size)
    )

    # decrypt
    key = PBKDF2(password, salt=b"salt", dkLen=32)
    decrypted = unpad(
        AES.new(key, AES.MODE_CBC, iv=ciphertext[: AES.block_size]).decrypt(
            ciphertext[AES.block_size :]
        ),
        AES.block_size,
    )

    assert plaintext == decrypted
