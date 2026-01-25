import os

# Thresholds for face matching
FACE_MATCH_THRESHOLD = 0.6  # Lower is stricter for dlib/face_recognition distance
MIN_CONFIDENCE_SCORE = 0.65  # Aggregated score required for login (Lowered for real-world use)

# Liveness Challenge Settings
CHALLENGE_TIMEOUT = 5.0  # Seconds to complete a liveness challenge
BLINK_THRESHOLD = 0.2    # Eye aspect ratio (EAR)
HEAD_MOVE_SENSITIVITY = 15.0 # Degrees

# Roles
ROLE_ADMIN = "admin"
ROLE_USER = "user"

# Security Rules
MAX_FACES_ALLOWED = 1

# Encryption Settings (In a real system, use ENV vars or Secret Manager)
# This is a sample key - DO NOT use in real production without changing!
ENCRYPTION_KEY = b'uF_p9d8j7G5h4k3L2m1n0o9p8q7r6s5t4u3v2w1x0y=' 

# Bootstrap Credentials (One-time use)
BOOTSTRAP_USER = "osamah"
BOOTSTRAP_PASS = "123456"

# System States
STATE_BOOTSTRAP = "BOOTSTRAP"
STATE_ACTIVE = "ACTIVE"
