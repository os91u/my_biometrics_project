import customtkinter as ctk
import cv2
from PIL import Image
import time
from config.gui_config import *

class EnrollmentDialog(ctk.CTkToplevel):
    def __init__(self, parent, controller, on_complete):
        super().__init__(parent)
        
        self.controller = controller
        self.on_complete = on_complete
        
        self.title("New User Enrollment")
        self.geometry("750x750")  # Increased height to show all elements
        self.resizable(False, False)
        
        # Make it modal
        self.transient(parent)
        self.grab_set()
        
        # Title
        self.title_label = ctk.CTkLabel(self, text="BIOMETRIC ENROLLMENT", font=FONT_TITLE)
        self.title_label.pack(pady=20)
        
        # Name Entry
        self.name_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.name_frame.pack(pady=10)
        
        ctk.CTkLabel(self.name_frame, text="Full Name:", font=FONT_BOLD).pack(side="left", padx=10)
        self.name_entry = ctk.CTkEntry(self.name_frame, width=300, height=40)
        self.name_entry.pack(side="left", padx=10)
        
        # Camera Feed
        self.cam_frame = ctk.CTkFrame(self, fg_color="#000000", corner_radius=10, width=640, height=360)
        self.cam_frame.pack(pady=10)
        self.cam_frame.pack_propagate(False)  # Maintain size
        self.cam_label = ctk.CTkLabel(self.cam_frame, text="")
        self.cam_label.pack(expand=True, fill="both")
        
        # Instructions
        self.instr_label = ctk.CTkLabel(self, text="Enter name and click Start to begin capture", 
                                        font=FONT_MAIN, text_color=COLOR_TEXT_DIM)
        self.instr_label.pack(pady=10)
        
        # Progress Bar
        self.progress = ctk.CTkProgressBar(self, width=400, progress_color=COLOR_PRIMARY)
        self.progress.pack(pady=10)
        self.progress.set(0)
        
        # Buttons - CRITICAL: These must be visible!
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=20, side="bottom")  # Force to bottom
        
        self.start_btn = ctk.CTkButton(self.btn_frame, text="Start Capture", 
                                       fg_color=COLOR_PRIMARY, hover_color="#1a4574",
                                       width=150, height=45, command=self.start_capture)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        self.cancel_btn = ctk.CTkButton(self.btn_frame, text="Cancel", 
                                        fg_color=COLOR_ERROR, hover_color="#c0392b",
                                        width=150, height=45, command=self.destroy)
        self.cancel_btn.grid(row=0, column=1, padx=10)
        
        self.is_capturing = False
        self.frames = []
        self.update_feed()
    
    def update_feed(self):
        """Shows live camera feed."""
        if not self.winfo_exists():
            return
            
        frame, _ = self.controller.get_ui_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((640, 360))
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 360))
            self.cam_label.configure(image=img_tk)
            self.cam_label.image = img_tk
        
        self.after(30, self.update_feed)
    
    def start_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            self.instr_label.configure(text="Please enter a name!", text_color=COLOR_ERROR)
            return
        
        self.is_capturing = True
        self.start_btn.configure(state="disabled")
        self.name_entry.configure(state="disabled")
        self.instr_label.configure(text="Capturing... Stay still and look at the camera", 
                                   text_color=COLOR_SECONDARY)
        
        self.frames = []
        self.capture_loop(name)
    
    def capture_loop(self, name):
        if not self.is_capturing or not self.winfo_exists():
            return
        
        frame, _ = self.controller.get_ui_frame()
        if frame is not None and len(self.frames) < 10:
            self.frames.append(frame.copy())
            progress_val = len(self.frames) / 10.0
            self.progress.set(progress_val)
            self.instr_label.configure(text=f"Capturing frame {len(self.frames)}/10...")
            self.after(300, lambda: self.capture_loop(name))
        elif len(self.frames) >= 10:
            self.finalize_enrollment(name)
    
    def finalize_enrollment(self, name):
        self.instr_label.configure(text="Processing biometric data...", text_color=COLOR_ACCENT)
        result = self.controller.attempt_enrollment(name, self.frames)
        
        if result['status'] == 'SUCCESS':
            self.instr_label.configure(text=f"✓ {name} enrolled successfully!", text_color=COLOR_SECONDARY)
            self.after(2000, lambda: self.on_complete(result))
            self.after(2000, self.destroy)
        else:
            self.instr_label.configure(text=f"✗ Enrollment failed: {result.get('message', '')}", 
                                       text_color=COLOR_ERROR)
            self.start_btn.configure(state="normal")
            self.name_entry.configure(state="normal")
            self.is_capturing = False
            self.progress.set(0)
