from core.face_engine import FaceEngine
from core.liveness_engine import LivenessEngine
from core.confidence_engine import ConfidenceEngine
from storage.face_repository import FaceRepository
from storage.audit_repository import AuditRepository
from auth.access_control import AccessControl
from config.security_config import MAX_FACES_ALLOWED

class AuthFlow:
    """Orchestrates the login and enrollment processes."""
    
    def __init__(self):
        self.face_repo = FaceRepository()
        self.audit_repo = AuditRepository()
        self.face_engine = FaceEngine()
        self.liveness_engine = LivenessEngine()
        self.confidence_engine = ConfidenceEngine()
        self.access_control = AccessControl(self.face_repo)
        
        self.current_user = None

    def login_attempt(self, frame, liveness_data, challenge, challenge_passed):
        """Handles a single login attempt."""
        print(f"DEBUG: login_attempt called. challenge_passed={challenge_passed}", flush=True)
        
        face_locations = self.face_engine.detect_faces(frame)
        num_faces = len(face_locations)
        
        print(f"DEBUG: Detected {num_faces} face(s)", flush=True)
        
        if num_faces == 0:
            return {"status": "NO_FACE", "message": "No face detected"}
            
        if num_faces > MAX_FACES_ALLOWED:
            self.audit_repo.log("unknown", "none", "LOGIN", "FAILED", f"Multiple faces ({num_faces})")
            return {"status": "MULTI_FACE", "message": f"Multiple faces detected ({num_faces})"}

        # Exactly 1 face
        encoding = self.face_engine.get_encodings(frame, face_locations)[0]
        
        # Check against all known users
        all_users = self.face_repo.get_all_users()
        print(f"DEBUG: Comparing against {len(all_users)} users in database: {list(all_users.keys())}", flush=True)
        
        best_user = None
        best_score = 0.0
        best_reason = "No match"
        
        for name, data in all_users.items():
            distances = self.face_engine.compare_faces(data['encodings'], encoding)
            score, reason = self.confidence_engine.calculate_login_score(distances, liveness_data, challenge_passed)
            
            print(f"DEBUG: User '{name}' -> score={score:.3f}, reason={reason}", flush=True)
            
            if score > best_score:
                best_score = score
                best_user = {"name": name, "role": data['role']}
                best_reason = reason
        
        print(f"DEBUG: Best match: {best_user}, score={best_score:.3f}", flush=True)
        
        if best_user and self.confidence_engine.is_access_granted(best_score):
            self.current_user = best_user
            self.audit_repo.log(best_user['name'], best_user['role'], "LOGIN", "SUCCESS")
            print(f"DEBUG: ACCESS GRANTED for {best_user['name']}", flush=True)
            return {"status": "SUCCESS", "user": best_user}
        else:
            self.audit_repo.log("unknown", "none", "LOGIN", "FAILED", best_reason)
            print(f"DEBUG: ACCESS DENIED - {best_reason}", flush=True)
            return {"status": "FAILED", "message": best_reason}

    def enroll_user(self, name, encodings):
        """Handles user enrollment."""
        if not self.access_control.can_enroll(self.current_user):
            self.audit_repo.log(self.current_user['name'] if self.current_user else "unknown", 
                                "none", "FACE_ENROLL", "DENIED", "Unauthorized")
            return {"status": "DENIED", "message": "Admin authorization required"}
            
        role = self.access_control.get_role_for_new_user()
        self.face_repo.save_user(name, role, encodings)
        self.audit_repo.log(name, role, "FACE_ENROLL", "SUCCESS")
        return {"status": "SUCCESS", "message": f"User {name} enrolled as {role}"}

    def logout(self):
        if self.current_user:
            self.audit_repo.log(self.current_user['name'], self.current_user['role'], "LOGOUT", "SUCCESS")
            self.current_user = None
