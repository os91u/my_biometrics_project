import customtkinter as ctk
from config.gui_config import *

class UserDashboard(ctk.CTkFrame):
    """Simple dashboard for regular users (non-admin)."""
    
    def __init__(self, master, controller, user, logout_callback):
        super().__init__(master, fg_color=COLOR_BG_DARK)
        self.controller = controller
        self.user = user
        self.logout_callback = logout_callback
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(self, fg_color=COLOR_BG_LIGHT, corner_radius=15)
        header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        title = ctk.CTkLabel(header_frame, text="🎉 USER DASHBOARD", font=FONT_TITLE, text_color=COLOR_PRIMARY)
        title.pack(pady=20)
        
        # Welcome Message
        welcome_frame = ctk.CTkFrame(self, fg_color=COLOR_BG_LIGHT, corner_radius=15)
        welcome_frame.grid(row=1, column=0, padx=40, pady=20, sticky="nsew")
        
        welcome_label = ctk.CTkLabel(welcome_frame, 
                                     text=f"Welcome, {user['name']}!", 
                                     font=("Inter", 32, "bold"),
                                     text_color=COLOR_SECONDARY)
        welcome_label.pack(pady=(40, 20))
        
        info_label = ctk.CTkLabel(welcome_frame,
                                 text=f"Role: {user['role'].upper()}\n\n✅ Login successful!\n\nThe navigation logic is working correctly.",
                                 font=FONT_MAIN,
                                 text_color=COLOR_TEXT_WHITE,
                                 justify="center")
        info_label.pack(pady=20)
        
        # Success indicator
        success_icon = ctk.CTkLabel(welcome_frame,
                                   text="✓",
                                   font=("Inter", 80, "bold"),
                                   text_color=COLOR_SECONDARY)
        success_icon.pack(pady=30)
        
        # Logout button
        logout_btn = ctk.CTkButton(self,
                                  text="Logout",
                                  font=FONT_BOLD,
                                  fg_color=COLOR_ERROR,
                                  hover_color="#c0392b",
                                  command=self.logout)
        logout_btn.grid(row=2, column=0, pady=20)
    
    def logout(self):
        self.logout_callback()
