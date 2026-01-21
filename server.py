import cv2
import json
import asyncio
from fastapi import FastAPI, Request, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.app_controller import AppController
from config.security_config import STATE_BOOTSTRAP, STATE_ACTIVE
import time

app = FastAPI(title="Face Recognition Biometric System")
controller = AppController()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Camera Streaming Logic ---
def generate_frames():
    """Generator for MJPEG camera stream with UI overlays."""
    controller.start_camera()
    while True:
        frame, liveness_data = controller.get_ui_frame()
        if frame is None:
            break

        # Apply basic status overlays for the web stream
        # (The frontend will also draw richer UI on top of this)
        status = "Scanning..."
        if controller.active_challenge:
            status = f"CHALLENGE: {controller.active_challenge}"
            if controller.challenge_met:
                status += " (OK!)"

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/api/stream")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# --- Authentication Endpoints ---

@app.get("/api/system/state")
async def get_state():
    return {"state": controller.get_system_state()}

@app.post("/api/auth/bootstrap")
async def bootstrap(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    if controller.verify_bootstrap(username, password):
        return {"status": "SUCCESS", "message": "Bootstrap verified. Proceed to enrollment."}
    return {"status": "FAILED", "message": "Invalid bootstrap credentials"}

@app.post("/api/auth/complete-bootstrap")
async def complete_bootstrap(request: Request):
    data = await request.json()
    username = data.get("username")
    # In web version, we capture frames from the stream or a high-res burst
    # For now, we reuse the controller's burst capture for simplicity
    controller.start_camera()
    time.sleep(1) # Wait for camera
    
    frames = []
    for _ in range(5):
        f, _ = controller.get_ui_frame()
        if f is not None:
            frames.append(f)
        time.sleep(0.5)
        
    result = controller.complete_bootstrap(username, frames)
    return result

@app.post("/api/auth/login-start")
async def login_start():
    challenge = controller.start_new_challenge()
    return {"challenge": challenge}

@app.post("/api/auth/login-attempt")
async def login_attempt():
    frame, liveness_data = controller.get_ui_frame()
    if frame is None:
        return {"status": "FAILED", "message": "Camera not ready"}
        
    result = controller.attempt_login(frame, liveness_data)
    return result

@app.post("/api/admin/enroll")
async def enroll_new_user(request: Request):
    data = await request.json()
    name = data.get("name")
    
    # Check if admin is logged in (simplified for first version)
    if not controller.auth_flow.current_user:
        return {"status": "DENIED", "message": "Admin login required"}
        
    # Capture burst
    frames = []
    for _ in range(5):
        f, _ = controller.get_ui_frame()
        if f is not None:
            frames.append(f)
        time.sleep(0.5)
        
    result = controller.attempt_enrollment(name, frames)
    return result

@app.get("/api/admin/logs")
async def get_logs():
    return controller.auth_flow.audit_repo.get_all_logs()

@app.post("/api/auth/logout")
async def logout():
    controller.logout()
    return {"status": "SUCCESS"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
