import cv2
import time
from config.system_config import WINDOW_NAME

class AdminUI:
    """Presentation layer for Admin tasks (Enrollment/Logs)."""
    
    def __init__(self, controller):
        self.controller = controller

    def run(self):
        while True:
            print("\n--- Admin Panel ---")
            print("1. Enroll New User")
            print("2. View Audit Logs")
            print("3. Logout")
            choice = input("Select option: ").lower()

            if choice == '1':
                self.enrollment_flow()
            elif choice == '2':
                self.show_logs()
            elif choice == '3' or choice == 'q':
                self.controller.logout()
                return "LOGIN"

    def enrollment_flow(self):
        name = input("Enter name for new user: ")
        print("Capturing frames... look at the camera and move slightly.")
        
        frames = []
        start_time = time.time()
        while len(frames) < 5 and (time.time() - start_time < 10):
            frame, _ = self.controller.get_ui_frame()
            if frame is not None:
                # Simple visual feedback
                cv2.putText(frame, f"Capturing... {len(frames)}/5", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow(WINDOW_NAME, frame)
                cv2.waitKey(1)
                frames.append(frame.copy())
                time.sleep(0.5)

        result = self.controller.attempt_enrollment(name, frames)
        print(f"Enrollment Result: {result['status']} - {result.get('message', '')}")

    def show_logs(self):
        logs = self.controller.auth_flow.audit_repo.get_logs()
        print("\n--- Audit Logs ---")
        for log in logs[-10:]: # Show last 10
            print(f"[{log['time']}] {log['action']} - {log['name']} ({log['role']}) -> {log['result']} {log.get('reason', '')}")
