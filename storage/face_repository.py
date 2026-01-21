import numpy as np
from storage.secure_storage import SecureStorage
from config.system_config import FACE_DATA_PATH

class FaceRepository:
    """Manages face encodings storage using SecureStorage abstraction."""
    
    def __init__(self):
        self.storage = SecureStorage()
        self.path = FACE_DATA_PATH
        self._data = self._load()

    def _load(self):
        """Loads data from encrypted file."""
        raw_data = self.storage.load_from_file(self.path)
        # Convert lists back to numpy arrays
        for user in raw_data.values():
            user['encodings'] = [np.array(e) for e in user['encodings']]
        return raw_data

    def save_user(self, name: str, role: str, encodings: list):
        """Saves a new user with their encodings."""
        # Convert numpy arrays to lists for JSON serialization
        enc_list = [e.tolist() for e in encodings]
        self._data[name] = {
            "role": role,
            "encodings": enc_list
        }
        self.storage.save_to_file(self.path, self._data)
        # Refresh local cache
        self._data = self._load()

    def get_user(self, name: str):
        return self._data.get(name)

    def get_all_users(self):
        return self._data

    def delete_user(self, name: str):
        if name in self._data:
            del self._data[name]
            self.storage.save_to_file(self.path, self._data)
            return True
        return False

    def is_empty(self):
        return len(self._data) == 0
