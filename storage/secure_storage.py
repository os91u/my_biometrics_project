import os
import json
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from config.security_config import ENCRYPTION_KEY

class SecureStorage:
    """Handles encryption and decryption of data at rest using AES-GCM."""
    
    def __init__(self):
        # In a real system, the key would be derived or fetched from a vault
        # For this implementation, we use a static key from config
        self.aesgcm = AESGCM(ENCRYPTION_KEY[:32]) # Use 256-bit key

    def encrypt(self, data: dict) -> bytes:
        """Encrypts dictionary data to bytes."""
        nonce = os.urandom(12)
        plaintext = json.dumps(data).encode('utf-8')
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, encrypted_data: bytes) -> dict:
        """Decrypts bytes to dictionary data."""
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        plaintext = self.aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext.decode('utf-8'))

    def save_to_file(self, path: str, data: dict):
        """Saves encrypted data to a file."""
        encrypted = self.encrypt(data)
        with open(path, 'wb') as f:
            f.write(encrypted)

    def load_from_file(self, path: str) -> dict:
        """Loads and decrypts data from a file."""
        if not os.path.exists(path):
            return {}
        with open(path, 'rb') as f:
            encrypted = f.read()
        return self.decrypt(encrypted)

    def get_system_state(self, path: str):
        """Retrieves the current system state."""
        from config.security_config import STATE_BOOTSTRAP
        data = self.load_from_file(path)
        return data.get("state", STATE_BOOTSTRAP)

    def set_system_state(self, path: str, state: str):
        """Sets the system state. Should be immutable once ACTIVE."""
        from config.security_config import STATE_ACTIVE
        current = self.get_system_state(path)
        if current == STATE_ACTIVE:
            return False # Cannot go back to BOOTSTRAP
        
        self.save_to_file(path, {"state": state})
        return True
