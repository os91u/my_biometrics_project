import sys
from app.app_controller import AppController
from ui.login_ui import LoginUI
from ui.admin_ui import AdminUI

def main():
    controller = AppController()
    login_ui = LoginUI(controller)
    admin_ui = AdminUI(controller)

    current_state = "LOGIN"

    try:
        while current_state != "EXIT":
            if current_state == "LOGIN":
                next_state = login_ui.run()
                if next_state == "ADMIN_PANEL":
                    current_state = "ADMIN_PANEL"
                elif next_state == "USER_DASHBOARD":
                    print("User dashboard (Not implemented) - Logging out.")
                    controller.logout()
                    current_state = "LOGIN"
                else:
                    current_state = "EXIT"

            elif current_state == "ADMIN_PANEL":
                current_state = admin_ui.run()

    except KeyboardInterrupt:
        print("\nSystem shutting down.")
    finally:
        controller.stop_camera()
        print("Goodbye.")

if __name__ == "__main__":
    main()
