from auth.auth_flow import AuthFlow
from auth.bootstrap_flow import BootstrapFlow
from core.camera_service import CameraService
from config.system_config import SYSTEM_STATE_PATH
from config.security_config import STATE_BOOTSTRAP, STATE_ACTIVE
import time

class AppController:
    """The main brain of the application, connecting UI to Auth."""
    
    def __init__(self):
        self.auth_flow = AuthFlow()
        self.camera = CameraService()
        self.bootstrap_flow = BootstrapFlow(self.auth_flow.face_repo, self.auth_flow.face_repo.storage)
        
        self.active_challenge = None
        self.challenge_start_time = 0
        self.challenge_met = False

    def get_system_state(self):
        """Checks if the system is in BOOTSTRAP or ACTIVE mode."""
        return self.auth_flow.face_repo.storage.get_system_state(SYSTEM_STATE_PATH)

    def verify_bootstrap(self, username, password):
        """Attempts bootstrap login."""
        return self.bootstrap_flow.verify_credentials(username, password)

    def complete_bootstrap(self, name, frames):
        """Enrolls first admin and activates system."""
        # Capture encodings
        encodings = []
        for f in frames:
            enc = self.auth_flow.face_engine.get_encodings(f)
            if enc:
                encodings.append(enc[0])
        
        if not encodings:
            return {"status": "FAILED", "message": "No stable face detected"}

        # Enroll user via bootstrap flow
        success = self.bootstrap_flow.complete_bootstrap(name, encodings)
        if success:
            # Switch system state to ACTIVE
            self.auth_flow.face_repo.storage.set_system_state(SYSTEM_STATE_PATH, STATE_ACTIVE)
            return {"status": "SUCCESS", "message": "System Activated. Admin enrolled."}
        
        return {"status": "FAILED", "message": "Bootstrap enrollment failed"}

    def start_camera(self):
        self.camera.start()

    def stop_camera(self):
        self.camera.stop()

    def get_ui_frame(self):
        """Returns the frame and status info for the UI to display."""
        frame = self.camera.get_frame()
        if frame is None:
            return None, {}

        # Basic liveness data for every frame
        liveness_data = self.auth_flow.liveness_engine.check_liveness(frame)
        
        # Check if challenge is met
        if self.active_challenge:
            if self.auth_flow.liveness_engine.verify_challenge(self.active_challenge, liveness_data):
                self.challenge_met = True
            
            # Timeout check
            if time.time() - self.challenge_start_time > 5.0: # Check config
                self.active_challenge = None # Expire
        
        return frame, liveness_data

    def start_new_challenge(self):
        self.active_challenge = self.auth_flow.liveness_engine.generate_challenge()
        self.challenge_start_time = time.time()
        self.challenge_met = False
        return self.active_challenge

    def attempt_login(self, frame, liveness_data):
        result = self.auth_flow.login_attempt(
            frame, 
            liveness_data, 
            self.active_challenge, 
            self.challenge_met
        )
        # Reset challenge after attempt
        self.active_challenge = None
        return result

    def attempt_enrollment(self, name, frames):
        """Processes multiple frames for enrollment."""
        encodings = []
        for f in frames:
            enc = self.auth_flow.face_engine.get_encodings(f)
            if enc:
                encodings.append(enc[0])
        
        if not encodings:
            return {"status": "FAILED", "message": "No stable face detected during enrollment"}
            
        return self.auth_flow.enroll_user(name, encodings)

    def get_current_user(self):
        return self.auth_flow.current_user
        
    def logout(self):
        self.auth_flow.logout()
