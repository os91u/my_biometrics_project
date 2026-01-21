import os

# Path Definitions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "data")
FACE_DATA_PATH = os.path.join(STORAGE_DIR, "faces.enc")
AUDIT_LOG_PATH = os.path.join(STORAGE_DIR, "audit.json")
SYSTEM_STATE_PATH = os.path.join(STORAGE_DIR, "state.enc")

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# UI Settings
WINDOW_NAME = "Secure Face Authentication"
