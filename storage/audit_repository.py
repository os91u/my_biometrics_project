import json
import os
from datetime import datetime
from config.system_config import AUDIT_LOG_PATH

class AuditRepository:
    """Manages secure, append-only security logs."""
    
    def __init__(self):
        self.log_path = AUDIT_LOG_PATH
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def log(self, name: str, role: str, action: str, result: str, reason: str = ""):
        """Appends a new security event to the audit log."""
        event = {
            "name": name,
            "role": role,
            "action": action,
            "result": result,
            "reason": reason,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.log_path, 'r+') as f:
                logs = json.load(f)
                logs.append(event)
                f.seek(0)
                json.dump(logs, f, indent=4)
                f.truncate()
        except Exception as e:
            # In production, we would use a more robust fallback like syslog
            print(f"CRITICAL: Failed to write audit log: {e}")

    def get_logs(self):
        """Retrieves all logs (Admin only access should be enforced by service)."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, 'r') as f:
            return json.load(f)
