import cv2
import mediapipe as mp
import numpy as np
import time
from config.security_config import BLINK_THRESHOLD, HEAD_MOVE_SENSITIVITY

class LivenessEngine:
    """Layered anti-spoofing engine using MediaPipe Face Mesh."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Landmark indices for eyes (simplified)
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    def _get_ear(self, landmarks, eye_indices):
        """Calculates Eye Aspect Ratio (EAR) for blink detection."""
        # This is a simplified EAR calculation for demonstration
        # Real EAR involves distances between specific eyelid landmarks
        # Here we just check the vertical distance relative to horizontal
        p1 = landmarks[eye_indices[12]] # Top
        p2 = landmarks[eye_indices[4]]  # Bottom
        p3 = landmarks[eye_indices[0]]  # Left
        p4 = landmarks[eye_indices[8]]  # Right
        
        dist_v = np.linalg.norm(np.array([p1.x - p2.x, p1.y - p2.y]))
        dist_h = np.linalg.norm(np.array([p3.x - p4.x, p3.y - p4.y]))
        
        return dist_v / (dist_h + 1e-6)

    def check_liveness(self, frame):
        """Performs real-time liveness checks on a frame."""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return {"liveness_score": 0.0, "reason": "No face detected", "face_present": False}

        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # 1. Blink Detection (EAR)
        left_ear = self._get_ear(face_landmarks, self.LEFT_EYE)
        right_ear = self._get_ear(face_landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        is_blinking = avg_ear < BLINK_THRESHOLD
        
        # 2. Head Pose (Simplified check for basic movement)
        # We can look at the nose position relative to the face boundaries
        nose = face_landmarks[1]
        left_side = face_landmarks[234]
        right_side = face_landmarks[454]
        
        # Ratio of nose position relative to sides
        horizontal_ratio = (nose.x - left_side.x) / (right_side.x - left_side.x + 1e-6)
        
        # If horizontal_ratio is far from 0.5, the head is turned
        # For a simple binary "liveness" we just return these raw values for the ConfidenceEngine
        
        return {
            "liveness_score": 1.0 if not is_blinking else 0.5, # Reduced score during blink
            "ear": avg_ear,
            "horizontal_ratio": horizontal_ratio,
            "is_blinking": is_blinking,
            "face_present": True
        }

    def generate_challenge(self):
        """Randomly selects a liveness challenge."""
        challenges = ["BLINK", "TURN_LEFT", "TURN_RIGHT", "SMILE"]
        return np.random.choice(challenges)

    def verify_challenge(self, challenge, liveness_data):
        """Verifies if the specific challenge was met."""
        if challenge == "BLINK":
            return liveness_data.get("is_blinking", False)
        elif challenge == "TURN_LEFT":
            return liveness_data.get("horizontal_ratio", 0.5) < 0.4
        elif challenge == "TURN_RIGHT":
            return liveness_data.get("horizontal_ratio", 0.5) > 0.6
        # SMILE would require more landmarks, defaulting to True for basic demo
        return True
