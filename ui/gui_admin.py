import customtkinter as ctk
from config.gui_config import *
from ui.gui_enrollment import EnrollmentDialog

class AdminDashboard(ctk.CTkFrame):
    def __init__(self, master, controller, user, on_logout):
        super().__init__(master, fg_color=COLOR_BG_DARK)
        self.controller = controller
        self.user_data = user
        self.on_logout = on_logout
        self.master_window = master

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        self.sidebar = ctk.CTkFrame(self, fg_color=COLOR_BG_LIGHT, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="NEURA ADMIN", font=FONT_BOLD)
        self.logo_label.pack(pady=20, padx=20)

        self.btn_users = ctk.CTkButton(self.sidebar, text="Enrolled Users", fg_color="transparent", 
                                        text_color=COLOR_TEXT_DIM, hover_color=COLOR_BG_DARK,
                                        command=self.show_users)
        self.btn_users.pack(fill="x", padx=10, pady=5)

        self.btn_logs = ctk.CTkButton(self.sidebar, text="Audit Logs", fg_color="transparent",
                                       text_color=COLOR_TEXT_DIM, hover_color=COLOR_BG_DARK,
                                       command=self.show_logs)
        self.btn_logs.pack(fill="x", padx=10, pady=5)

        self.spacer = ctk.CTkLabel(self.sidebar, text="")
        self.spacer.pack(expand=True)

        self.btn_logout = ctk.CTkButton(self.sidebar, text="LOGOUT", fg_color=COLOR_ERROR, 
                                         hover_color="#c0392b", command=self.on_logout)
        self.btn_logout.pack(fill="x", padx=10, pady=20)

        # --- Main View Container ---
        self.main_view = ctk.CTkFrame(self, fg_color="transparent")
        self.main_view.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.show_users()

    def show_users(self):
        self.clear_main()
        title = ctk.CTkLabel(self.main_view, text="User Management", font=FONT_TITLE)
        title.pack(anchor="w", pady=(0, 20))

        # Add User Button
        self.add_btn = ctk.CTkButton(self.main_view, text="+ Enroll New User", fg_color=COLOR_PRIMARY,
                                      command=self.enroll_procedure)
        self.add_btn.pack(anchor="w", pady=(0, 10))

        # Table-like display
        users = self.controller.auth_flow.face_repo.get_all_users()
        scroll_frame = ctk.CTkScrollableFrame(self.main_view, fg_color=COLOR_BG_LIGHT)
        scroll_frame.pack(fill="both", expand=True)

        for name, data in users.items():
            user_row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            user_row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(user_row, text=name, font=FONT_BOLD, width=200, anchor="w").pack(side="left", padx=10)
            ctk.CTkLabel(user_row, text=data['role'].upper(), text_color=COLOR_ACCENT).pack(side="left", padx=10)

    def show_logs(self):
        self.clear_main()
        title = ctk.CTkLabel(self.main_view, text="Security Audit Logs", font=FONT_TITLE)
        title.pack(anchor="w", pady=(0, 20))

        logs = self.controller.auth_flow.audit_repo.get_all_logs()
        log_box = ctk.CTkTextbox(self.main_view, fg_color=COLOR_BG_LIGHT, font=("Consolas", 12))
        log_box.pack(fill="both", expand=True)
        
        if logs:
            formatted_lines = []
            for l in reversed(logs):
                # Handle both old and new log formats
                timestamp = l.get('timestamp') or l.get('time', 'N/A')
                user = l.get('user') or l.get('name', 'unknown')
                event = l.get('event') or l.get('action', 'N/A')
                role = l.get('role', 'N/A')
                status = l.get('status') or l.get('result', 'N/A')
                message = l.get('message') or l.get('reason', '')
                
                line = f"[{timestamp}] {event} - {user} ({role}) -> {status}"
                if message:
                    line += f" | {message}"
                formatted_lines.append(line)
            
            formatted_logs = "\n".join(formatted_lines)
        else:
            formatted_logs = "No audit logs available yet."
            
        log_box.insert("0.0", formatted_logs)
        log_box.configure(state="disabled")

    def enroll_procedure(self):
        """Opens the enrollment dialog."""
        EnrollmentDialog(self.master_window, self.controller, self.on_enrollment_complete)

    def on_enrollment_complete(self, result):
        """Callback after enrollment finishes."""
        self.show_users()  # Refresh the user list

    def clear_main(self):
        for widget in self.main_view.winfo_children():
            widget.destroy()
