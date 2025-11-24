# face_logic.py (Version 2.0 - Multi-Angle Enrollment)

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
#  "http://172.17.1.19:8080/video"
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 2
# عدد البصمات التي سنأخذها لكل شخص
NUM_ENCODINGS_PER_PERSON = 3 

# --- دوال مساعدة (تبقى كما هي) ---
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

# --- الوظائف الرئيسية (المحسنة) ---

def add_new_face_enhanced():
    """
    إضافة بصمات وجه جديدة من زوايا متعددة.
    """
    data = load_encodings()
    vs = cv2.VideoCapture(CAMERA_SOURCE)
    
    # طلب اسم الشخص أولاً
    new_name = simpledialog.askstring("Input", "Please enter the name for the new person:", parent=None)
    if not new_name or not new_name.strip():
        messagebox.showwarning("Cancelled", "Name input was cancelled or empty.")
        return

    captured_encodings = []
    instructions = [
        "1. Look STRAIGHT at the camera and BLINK.",
        "2. Slowly turn your head to the RIGHT.",
        "3. Slowly turn your head to the LEFT."
    ]

    for i, instruction in enumerate(instructions):
        messagebox.showinfo(f"Step {i+1}/{len(instructions)}", instruction)
        face_captured_for_step = False
        
        while not face_captured_for_step:
            ret, frame = vs.read()
            if not ret: break
            
            # عرض التعليمات على الشاشة
            cv2.putText(frame, f"Step {i+1}: {instruction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Enrollment Process (Press 's' to capture, 'q' to quit)", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # السماح للمستخدم بالتقاط الصورة يدوياً بالضغط على 's'
            if key == ord('s'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # كشف الحيوية (الرمش) مطلوب فقط للخطوة الأولى
                is_live = False
                if i == 0:
                    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
                    for face_landmarks in face_landmarks_list:
                        ear = (eye_aspect_ratio(face_landmarks['left_eye']) + eye_aspect_ratio(face_landmarks['right_eye'])) / 2.0
                        if ear < EYE_ASPECT_RATIO_THRESHOLD:
                            is_live = True
                            break
                else:
                    is_live = True # نتجاوز كشف الحيوية للقطات الجانبية

                if is_live:
                    encodings = face_recognition.face_encodings(rgb_frame)
                    if encodings:
                        captured_encodings.append(encodings[0])
                        print(f"[INFO] Captured encoding #{len(captured_encodings)} for {new_name}.")
                        face_captured_for_step = True
                    else:
                        messagebox.showwarning("Warning", "Could not find a face. Please try again.")
                else:
                    messagebox.showwarning("Warning", "Liveness check failed (Blink not detected). Please try again.")

            elif key == ord('q'):
                vs.release()
                cv2.destroyAllWindows()
                messagebox.showerror("Error", "Enrollment cancelled by user.")
                return

    vs.release()
    cv2.destroyAllWindows()

    if len(captured_encodings) == len(instructions):
        for encoding in captured_encodings:
            data["encodings"].append(encoding)
            data["names"].append(new_name)
        
        save_encodings(data)
        messagebox.showinfo("Success", f"Successfully captured {len(captured_encodings)} face signatures for '{new_name}'.")
    else:
        messagebox.showerror("Error", "Failed to capture all required face signatures. Please try again.")


def verify_face():
    """
    التحقق من بصمة الوجه (الكود يبقى كما هو لأن منطق المقارنة يعالجه تلقائياً).
    face_recognition.compare_faces سيقارن البصمة الجديدة مع كل البصمات في قاعدة البيانات.
    """
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
