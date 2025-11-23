# face_logic.py

import face_recognition
import pickle
import cv2
from scipy.spatial import distance as dist
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# --- إعدادات ---
ENCODINGS_FILE = "encodings.pickle"
CAMERA_SOURCE = 0
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 2

# --- دوال مساعدة ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def load_encodings():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"encodings": [], "names": []}
    return data

def save_encodings(data):
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

# --- الوظائف الرئيسية ---
def add_new_face():
    data = load_encodings()
    vs = cv2.VideoCapture(CAMERA_SOURCE)
    
    frame_counter = 0
    face_captured = False
    
    messagebox.showinfo("Add New Face", "A camera window will open. Please look at the camera and BLINK to capture your face signature.")
    
    while not face_captured:
        ret, frame = vs.read()
        if not ret: break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        
        is_blinking = False
        for face_landmarks in face_landmarks_list:
            ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
            if ear < EYE_ASPECT_RATIO_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= EYE_ASPECT_RATIO_CONSEC_FRAMES: is_blinking = True
                frame_counter = 0
        
        if is_blinking:
            messagebox.showinfo("Success", "Liveness confirmed (Blink detected)!")
            encodings = face_recognition.face_encodings(rgb_frame)
            if encodings:
                new_encoding = encodings[0]
                
                # استخدام نافذة منبثقة لإدخال الاسم
                new_name = simpledialog.askstring("Input", "Face signature captured. Please enter your name:", parent=None)
                
                if new_name and new_name.strip():
                    data["encodings"].append(new_encoding)
                    data["names"].append(new_name)
                    save_encodings(data)
                    messagebox.showinfo("Success", f"Face signature for '{new_name}' has been saved.")
                    face_captured = True
                else:
                    messagebox.showwarning("Cancelled", "Name input was cancelled. Please try again.")
                    break # الخروج من الحلقة إذا ألغى المستخدم الإدخال
            else:
                messagebox.showwarning("Warning", "Blink detected, but could not capture face encoding. Please try again.")

        cv2.imshow("Add New Face - BLINK to capture (Press 'q' to cancel)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
            
    vs.release()
    cv2.destroyAllWindows()

def verify_face():
    data = load_encodings()
    if not data["encodings"]:
        messagebox.showerror("Error", "No faces have been saved yet. Please add a face first.")
        return
        
    vs = cv2.VideoCapture(CAMERA_SOURCE)
    messagebox.showinfo("Verify Face", "Camera will start. Recognition begins after a blink.")

    is_live = False
    while True:
        ret, frame = vs.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if not is_live:
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            for face_landmarks in face_landmarks_list:
                ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    is_live = True
                    print("[INFO] Liveness confirmed. Starting recognition...")
        
        if is_live:
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            encodings = face_recognition.face_encodings(rgb_frame, boxes)
            names = []
            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding)
                name = "Unknown"
                if True in matches:
                    matched_idxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matched_idxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Face Verification (Press 'q' to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
            
    vs.release()
    cv2.destroyAllWindows()
