import customtkinter as ctk
import sys
from app.app_controller import AppController
from ui.gui_login import LoginView
from ui.gui_admin import AdminDashboard
from ui.gui_user_dashboard import UserDashboard
from ui.gui_bootstrap import BootstrapView
from config.gui_config import *
from config.security_config import STATE_BOOTSTRAP

class BiometricApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NEURA Biometric Security")
        self.geometry("1000x700")
        ctk.set_appearance_mode(THEME_DARK)
        
        self.controller = AppController()
        # Start camera immediately for the GUI
        self.controller.start_camera()
        
        # Initialize state
        self.current_frame = None
        
        # Check if we need to bootstrap
        if self.controller.get_system_state() == STATE_BOOTSTRAP:
            self.show_bootstrap()
        else:
            self.show_login()

    def show_bootstrap(self):
        if self.current_frame:
            self.current_frame.destroy()
        
        self.current_frame = BootstrapView(self, self.controller, self.show_login)
        self.current_frame.pack(fill="both", expand=True)

    def show_login(self):
        if self.current_frame:
            self.current_frame.destroy()
        
        self.current_frame = LoginView(self, self.controller, self.on_login_success)
        self.current_frame.pack(fill="both", expand=True)

    def on_login_success(self, user):
        print(f"DEBUG: on_login_success called with user: {user}", flush=True)
        if user['role'] == 'admin':
            print("DEBUG: Navigating to Admin Dashboard", flush=True)
            self.show_admin(user)
        else:
            print(f"DEBUG: Navigating to User Dashboard for {user['name']}", flush=True)
            self.show_user_dashboard(user)

    def show_admin(self, user):
        if self.current_frame:
            self.current_frame.destroy()
        
        self.current_frame = AdminDashboard(self, self.controller, user, self.logout)
        self.current_frame.pack(fill="both", expand=True)
    
    def show_user_dashboard(self, user):
        if self.current_frame:
            self.current_frame.destroy()
        
        self.current_frame = UserDashboard(self, self.controller, user, self.logout)
        self.current_frame.pack(fill="both", expand=True)

    def logout(self):
        self.controller.logout()
        self.show_login()

    def on_closing(self):
        self.controller.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = BiometricApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
