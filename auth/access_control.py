from config.security_config import ROLE_ADMIN, ROLE_USER

class AccessControl:
    """Enforces Role-Based Access Control (RBAC)."""
    
    def __init__(self, face_repo):
        self.face_repo = face_repo

    def can_enroll(self, current_user):
        """Only authenticated admins can enroll new faces."""
        if current_user and current_user.get('role') == ROLE_ADMIN:
            return True
        # If system is empty, first user is admin
        if self.face_repo.is_empty():
            return True
        return False

    def get_role_for_new_user(self):
        """First user is admin, others are base users."""
        if self.face_repo.is_empty():
            return ROLE_ADMIN
        return ROLE_USER

    def is_admin(self, user_role):
        return user_role == ROLE_ADMIN
