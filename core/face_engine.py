import face_recognition
import numpy as np
import cv2

class FaceEngine:
    """Core logic for face detection and encoding."""
    
    def detect_faces(self, frame):
        """Detects face locations in a frame."""
        # Convert BGR (OpenCV) to RGB (face_recognition) using cv2 for contiguous array
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        return face_locations

    def get_encodings(self, frame, face_locations=None):
        """Extracts facial encodings for detected faces."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if face_locations is None:
            face_locations = face_recognition.face_locations(rgb_frame)
        
        # Ensure contiguous array just in case
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        return encodings

    def compare_faces(self, known_encodings, face_encoding):
        """Compares a face encoding against a list of known encodings.
        Returns distances (lower is better).
        """
        if not known_encodings:
            return []
        
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        return distances
