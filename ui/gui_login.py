import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import time
from config.gui_config import *

class LoginView(ctk.CTkFrame):
    def __init__(self, master, controller, on_login_success):
        super().__init__(master, fg_color=COLOR_BG_DARK)
        self.controller = controller
        self.on_login_success = on_login_success
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self, text="NEURA ID LOGIN", font=FONT_TITLE, text_color=COLOR_TEXT_WHITE)
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        # Camera Container
        self.cam_frame = ctk.CTkFrame(self, fg_color="#000000", corner_radius=15, width=640, height=480)
        self.cam_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="")
        self.cam_label.pack(expand=True, fill="both")

        # Sidebar / Info Card (Glassmorphism style)
        self.info_card = ctk.CTkFrame(self, fg_color=COLOR_BG_LIGHT, corner_radius=15, width=250)
        self.info_card.grid(row=1, column=1, padx=(0, 20), pady=10, sticky="nsew")
        
        self.status_label = ctk.CTkLabel(self.info_card, text="SYSTEM STATUS", font=FONT_BOLD)
        self.status_label.pack(pady=(20, 10))
        
        self.status_val = ctk.CTkLabel(self.info_card, text="Waiting...", text_color=COLOR_SECONDARY, font=FONT_MAIN)
        self.status_val.pack(pady=5)

        self.liveness_label = ctk.CTkLabel(self.info_card, text="Liveness Index", font=FONT_BOLD)
        self.liveness_label.pack(pady=(20, 10))
        self.liveness_bar = ctk.CTkProgressBar(self.info_card, progress_color=COLOR_PRIMARY)
        self.liveness_bar.pack(padx=20, pady=5)
        self.liveness_bar.set(0)

        # Controls
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=2, column=0, columnspan=2, pady=20)

        self.challenge_btn = ctk.CTkButton(self.btn_frame, text="Start Challenge", font=FONT_BOLD, 
                                           fg_color=COLOR_PRIMARY, hover_color="#1a4574",
                                           command=self.start_challenge)
        self.challenge_btn.pack(side="left", padx=10)

        self.login_btn = ctk.CTkButton(self.btn_frame, text="Login with Face", font=FONT_BOLD,
                                       fg_color=COLOR_SECONDARY, hover_color="#27ae60",
                                       state="disabled", command=self.attempt_login)
        self.login_btn.pack(side="left", padx=10)

        self.is_running = True
        self.update_camera()

    def update_camera(self):
        if not self.is_running:
            return

        frame, liveness_data = self.controller.get_ui_frame()
        if frame is not None:
            # Check for face presence
            face_detected = liveness_data.get('face_present', False)
            
            # Update UI elements based on data
            status = "Please look at the camera"
            if face_detected:
                status = "Face Detected - Ready"
                if self.controller.active_challenge:
                    status = f"CHALLENGE: {self.controller.active_challenge}"
            
            if self.controller.challenge_met:
                status = "Liveness Verified! Click Login"
                self.login_btn.configure(state="normal", fg_color=COLOR_SECONDARY)
            else:
                self.login_btn.configure(state="disabled", fg_color="#34495e")
            
            self.status_val.configure(text=status)
            
            ear = liveness_data.get('ear', 0.0)
            # Normalize EAR (0.15 - 0.35) to 0-1
            progress = max(0, min(1, (ear - 0.15) / 0.2))
            self.liveness_bar.set(progress)

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Use PIL to resize and convert to PhotoImage for Tkinter
            img_pil = Image.fromarray(rgb_frame)
            img_pil = img_pil.resize((640, 480), Image.Resampling.LANCZOS)
            
            # Using CTkImage correctly
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            
            self.cam_label.configure(image=img_tk)
            self.cam_label.image = img_tk # Keep a reference
        else:
            if self.is_running:
                print("DEBUG: Frame is None in LoginView", flush=True)
                self.status_val.configure(text="Camera Waiting...", text_color=COLOR_TEXT_DIM)

        self.after(30, self.update_camera)

    def start_challenge(self):
        challenge = self.controller.start_new_challenge()
        # Visual feedback
        self.status_val.configure(text=f"DO: {challenge}")

    def attempt_login(self):
        print("DEBUG: attempt_login called in LoginView", flush=True)
        frame, liveness_data = self.controller.get_ui_frame()
        result = self.controller.attempt_login(frame, liveness_data)
        
        print(f"DEBUG: Login result in LoginView: {result}", flush=True)
        
        if result['status'] == 'SUCCESS':
            print(f"DEBUG: Login SUCCESS - User: {result.get('user')}", flush=True)
            self.status_val.configure(text="ACCESS GRANTED!", text_color=COLOR_SECONDARY)
            self.is_running = False
            # Call the transition immediately to see if there's any delay issue
            print("DEBUG: About to call on_login_success callback", flush=True)
            user_data = result['user']
            print(f"DEBUG: User data to pass: {user_data}", flush=True)
            # Use `after` to ensure the UI updates first, then transition
            self.after(1000, lambda: self._transition_to_admin(user_data))
        else:
            print(f"DEBUG: Login FAILED - {result.get('message')}", flush=True)
            self.status_val.configure(text=result.get('message', 'Login Failed'), text_color=COLOR_ERROR)
    
    def _transition_to_admin(self, user_data):
        print(f"DEBUG: _transition_to_admin called with user: {user_data}", flush=True)
        try:
            self.on_login_success(user_data)
            print("DEBUG: on_login_success callback executed successfully", flush=True)
        except Exception as e:
            print(f"ERROR: Exception in on_login_success: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def stop(self):
        self.is_running = False
