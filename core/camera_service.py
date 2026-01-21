import cv2
from config.system_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

class CameraService:
    """Handles camera initialization and frame capture."""
    
    def __init__(self):
        self.cap = None

    def start(self):
        """Starts the camera stream."""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise Exception("Could not open camera.")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def get_frame(self):
        """Captures a single frame."""
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def stop(self):
        """Releases the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.stop()
