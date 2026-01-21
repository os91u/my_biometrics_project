import customtkinter as ctk
import cv2
from PIL import Image
import time
from config.gui_config import *

class BootstrapView(ctk.CTkFrame):
    def __init__(self, master, controller, on_bootstrap_complete):
        super().__init__(master, fg_color=COLOR_BG_DARK)
        self.controller = controller
        self.on_bootstrap_complete = on_bootstrap_complete
        
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(self, text="SYSTEM INITIALIZATION", font=FONT_TITLE, text_color=COLOR_TEXT_WHITE)
        self.title_label.pack(pady=40)

        # Login Card
        self.card = ctk.CTkFrame(self, fg_color=COLOR_BG_LIGHT, corner_radius=20, width=400, height=350)
        self.card.pack(pady=20, padx=40)
        self.card.pack_propagate(False)

        self.info_text = ctk.CTkLabel(self.card, text="Enter the master credentials\nto activate your biometric shield.", 
                                      font=FONT_MAIN, text_color=COLOR_TEXT_DIM)
        self.info_text.pack(pady=(30, 20))

        self.user_entry = ctk.CTkEntry(self.card, placeholder_text="Username", width=300, height=45)
        self.user_entry.pack(pady=10)

        self.pass_entry = ctk.CTkEntry(self.card, placeholder_text="Password", show="*", width=300, height=45)
        self.pass_entry.pack(pady=10)

        self.msg_label = ctk.CTkLabel(self.card, text="", text_color=COLOR_ERROR, font=FONT_MAIN)
        self.msg_label.pack(pady=5)

        self.start_btn = ctk.CTkButton(self.card, text="AUTHENTICATE & START", font=FONT_BOLD, 
                                        fg_color=COLOR_PRIMARY, hover_color="#1a4574",
                                        width=300, height=50, command=self.handle_auth)
        self.start_btn.pack(pady=(20, 0))

    def handle_auth(self):
        user = self.user_entry.get()
        pw = self.pass_entry.get()
        
        if self.controller.verify_bootstrap(user, pw):
            self.start_enrollment(user)
        else:
            self.msg_label.configure(text="Access Denied: Invalid Credentials")

    def start_enrollment(self, username):
        # Switch to enrollment mode
        self.card.destroy()
        
        self.title_label.configure(text="ENROLLING ADMIN BIOMETRICS")
        
        self.cam_frame = ctk.CTkFrame(self, fg_color="#000000", corner_radius=15, width=640, height=480)
        self.cam_frame.pack(pady=20)
        
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="")
        self.cam_label.pack()

        self.instr_label = ctk.CTkLabel(self, text="Position your face in the center and stay still...", font=FONT_BOLD)
        self.instr_label.pack(pady=10)

        self.frames = []
        self.is_capturing = True
        self.username = username
        self.capture_loop()

    def capture_loop(self):
        if not self.is_capturing:
            return

        frame, _ = self.controller.get_ui_frame()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame).resize((640, 480))
            img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
            self.cam_label.configure(image=img_tk)
            self.cam_label.image = img_tk

            if len(self.frames) < 10:
                self.frames.append(frame.copy())
                self.instr_label.configure(text=f"Capturing Biometrics: {len(self.frames)*10}%")
            else:
                self.is_capturing = False
                self.finalize_bootstrap()
                return

        self.after(100, self.capture_loop)

    def finalize_bootstrap(self):
        self.instr_label.configure(text="Encrypting & Activating System...", text_color=COLOR_SECONDARY)
        result = self.controller.complete_bootstrap(self.username, self.frames)
        
        if result['status'] == 'SUCCESS':
            self.after(2000, self.on_bootstrap_complete)
        else:
            self.instr_label.configure(text="Activation Failed! Please restart.", text_color=COLOR_ERROR)
