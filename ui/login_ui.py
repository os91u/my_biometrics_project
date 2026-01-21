import cv2
import time
from config.system_config import WINDOW_NAME

class LoginUI:
    """Presentation layer for Authentication."""
    
    def __init__(self, controller):
        self.controller = controller

    def run(self):
        # 1. Check System Mode
        from config.security_config import STATE_BOOTSTRAP
        if self.controller.get_system_state() == STATE_BOOTSTRAP:
            return self.bootstrap_ui_flow()

        # 2. Normal Login Flow
        print("DEBUG: Starting camera...", flush=True)
        try:
            self.controller.start_camera()
        except Exception as e:
            print(f"CRITICAL ERROR: Camera startup failed: {e}", flush=True)
            return "EXIT"

        cv2.namedWindow(WINDOW_NAME)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        print("--- Secure Face Login System ---", flush=True)
        print("Press 'C' to start challenge, 'L' to Login, 'Q' to Quit", flush=True)

        iter_count = 0
        while True:
            iter_count += 1
            if iter_count % 10 == 0:
                print(f"DEBUG: UI Loop iteration {iter_count}", flush=True)
                
            frame, liveness_data = self.controller.get_ui_frame()
            if frame is None:
                print("DEBUG: Camera frame is None. retrying...", flush=True)
                time.sleep(0.1)
                frame, liveness_data = self.controller.get_ui_frame()
                if frame is None:
                    print("ERROR: Camera lost connection or busy.", flush=True)
                    break

            status_msg = "Waiting..."
            color = (255, 255, 255)

            # Display active challenge
            if self.controller.active_challenge:
                status_msg = f"CHALLENGE: {self.controller.active_challenge}"
                color = (0, 255, 255) # Yellow
                if self.controller.challenge_met:
                    status_msg += " (OK!) - Press 'L' to Login"
                    color = (0, 255, 0) # Green

            # Draw UI overlays on frame
            cv2.putText(frame, f"Status: {status_msg}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show EAR for visual feedback
            ear = liveness_data.get('ear', 0.0)
            cv2.putText(frame, f"Liveness: {ear:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key != 255: # If a key was actually pressed
                print(f"DEBUG: Key detected: {key} ('{chr(key) if 32 <= key <= 126 else '?'}')", flush=True)

            if key in [ord('q'), ord('Q')]:
                print("DEBUG: Quit key pressed.", flush=True)
                break
            elif key in [ord('c'), ord('C')]:
                challenge = self.controller.start_new_challenge()
                print(f"Challenge Started: {challenge}", flush=True)
            elif key in [ord('l'), ord('L')]:
                print("DEBUG: Login attempt triggered.", flush=True)
                if self.controller.challenge_met:
                    result = self.controller.attempt_login(frame, liveness_data)
                    print(f"Login Result: {result['status']} - {result.get('message', '')}", flush=True)
                    if result['status'] == 'SUCCESS':
                        print(f"Welcome {result['user']['name']} ({result['user']['role']})")
                        time.sleep(2)
                        cv2.destroyAllWindows()
                        if result['user']['role'] == 'admin':
                            return "ADMIN_PANEL"
                        return "USER_DASHBOARD"
                else:
                    print("Complete liveness challenge first!")

        self.controller.stop_camera()
        cv2.destroyAllWindows()
        return "EXIT"

    def bootstrap_ui_flow(self):
        """First-run initialization UI."""
        from config.system_config import WINDOW_NAME
        import cv2
        print("\n--- SYSTEM BOOTSTRAP ---")
        print("No users found. Please initialize the Admin account.")
        
        user = input("Enter Bootstrap Username: ")
        pw = input("Enter Bootstrap Password: ")
        
        if self.controller.verify_bootstrap(user, pw):
            print("Credentials Verified! Proceeding to Admin Face Enrollment.")
            time.sleep(1)
            
            self.controller.start_camera()
            print("Capturing Admin biometric data... Look at the camera.")
            
            frames = []
            start_time = time.time()
            while len(frames) < 5 and (time.time() - start_time < 15):
                frame, _ = self.controller.get_ui_frame()
                if frame is not None:
                    cv2.putText(frame, f"BOOTSTRAP: Capturing {len(frames)}/5", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow(WINDOW_NAME, frame)
                    cv2.waitKey(1)
                    frames.append(frame.copy())
                    time.sleep(0.5)
            
            result = self.controller.complete_bootstrap(user, frames)
            print(f"Bootstrap Result: {result['status']} - {result['message']}")
            
            self.controller.stop_camera()
            cv2.destroyAllWindows()
            
            if result['status'] == 'SUCCESS':
                print("System Activated! Please log in using your face now.")
                return "LOGIN"
        else:
            print("Access Denied: Invalid Bootstrap Credentials.")
            time.sleep(2)
        
        return "EXIT"
