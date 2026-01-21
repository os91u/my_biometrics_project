from config.security_config import BOOTSTRAP_USER, BOOTSTRAP_PASS, STATE_ACTIVE
from storage.audit_repository import AuditRepository

class BootstrapFlow:
    """Handles the first-run initialization (Bootstrap Mode)."""
    
    def __init__(self, face_repo, secure_storage):
        self.face_repo = face_repo
        self.storage = secure_storage
        self.audit_repo = AuditRepository()
        self.bootstrap_user = BOOTSTRAP_USER
        self.bootstrap_pass = BOOTSTRAP_PASS

    def verify_credentials(self, username, password):
        """Validates temporary bootstrap credentials."""
        # Note: In a real production system, this would involve a secure verify step
        # but for this bootstrap requirement, we use the specified credentials.
        if username == self.bootstrap_user and password == self.bootstrap_pass:
            return True
        
        self.audit_repo.log(username, "unknown", "BOOTSTRAP_LOGIN", "FAILED", "Invalid credentials")
        return False

    def complete_bootstrap(self, name, encodings):
        """Finalizes bootstrap by enrolling the admin and switching state."""
        # Enclose the admin enrollment
        from config.security_config import ROLE_ADMIN
        self.face_repo.save_user(name, ROLE_ADMIN, encodings)
        
        # Log the success
        self.audit_repo.log(name, ROLE_ADMIN, "SYSTEM_BOOTSTRAP", "SUCCESS", "Initial admin created")
        
        return True
