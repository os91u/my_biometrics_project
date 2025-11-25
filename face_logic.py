# ==============================================================================
# face_logic.py - THE DEFINITIVE, RELIABLE BUILD
#
# هذا هو الإصدار النهائي الذي يركز على الاستقرار والموثوقية.
# 1. إضافة وجه: موجهة يدوياً، بسيطة، وتتحقق فقط من وجود وجه واحد.
# 2. تحقق: آمن (حركة الرأس) وسريع (معالجة متقطعة).
# ==============================================================================

import face_recognition
import pickle
import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import random

# --- إعدادات رئيسية ---
ENCODINGS_FILE = "encodings.pickle"
CAMERA_SOURCE = 0 
LIVENESS_MOVEMENT_THRESHOLD = 50 
VERIFY_PROCESS_EVERY_N_FRAMES = 4 

# --- دوال مساعدة ---
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

# ==============================================================================
# 1. دالة إضافة وجه (بسيطة ومستقرة)
# ==============================================================================
def add_new_face_enhanced():
    """
    إضافة بصمات وجه جديدة بطريقة موجهة يدوياً. هذه الطريقة هي الأكثر استقراراً.
    النظام يثق في أن المستخدم يتبع التعليمات، ويتحقق فقط من وجود وجه واحد.
    """
    data = load_encodings()
    vs = cv2.VideoCapture(CAMERA_SOURCE)

    new_name = simpledialog.askstring("Input", "Please enter the name for the new person:", parent=None)
    if not new_name or not new_name.strip():
        messagebox.showwarning("Cancelled", "Name input was cancelled or empty.")
        vs.release()
        return

    captured_encodings = []
    instructions = [
        "1. Look STRAIGHT at the camera.",
        "2. Slowly turn your head to the RIGHT.",
        "3. Slowly turn your head to the LEFT."
    ]

    for instruction in instructions:
        messagebox.showinfo("Instructions", f"{instruction}\n\nPosition yourself, then press 's' in the camera window to capture.")
        
        face_captured_for_step = False
        while not face_captured_for_step:
            ret, frame = vs.read()
            if not ret: break

            cv2.putText(frame, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, "Press 's' to capture, 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Enrollment: Follow Instructions", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb_frame, model="hog")
                
                if len(boxes) == 1:
                    encoding = face_recognition.face_encodings(rgb_frame, boxes)[0]
                    captured_encodings.append(encoding)
                    print(f"[INFO] Capture successful! ({len(captured_encodings)}/3)")
                    face_captured_for_step = True # نجحت، انتقل للخطوة التالية
                elif len(boxes) > 1:
                    messagebox.showwarning("Warning", "Multiple faces detected. Please ensure only one person is in the frame.")
                else:
                    messagebox.showwarning("Warning", "No face detected. Please position yourself clearly.")

            elif key == ord('q'):
                vs.release()
                cv2.destroyAllWindows()
                messagebox.showerror("Error", "Enrollment cancelled.")
                return

    vs.release()
    cv2.destroyAllWindows()

    if len(captured_encodings) == len(instructions):
        for encoding in captured_encodings:
            data["encodings"].append(encoding)
            data["names"].append(new_name)
        save_encodings(data)
        messagebox.showinfo("Success", f"Enrollment complete for '{new_name}'.")
    else:
        messagebox.showerror("Error", "Enrollment failed. Please try again.")

# ==============================================================================
# 2. دالة التحقق من الوجه (آمنة وسريعة)
# ==============================================================================
def verify_face():
    """
    التحقق من بصمة الوجه مع كشف حيوية بحركة الرأس وأداء محسن.
    """
    data = load_encodings()
    if not data["encodings"]:
        messagebox.showerror("Error", "No faces saved. Please add a face first.")
        return

    vs = cv2.VideoCapture(CAMERA_SOURCE)
    
    challenge_direction = random.choice(["LEFT", "RIGHT"])
    challenge_text = f"Turn head {challenge_direction}"
    liveness_confirmed = False
    initial_face_x = None

    messagebox.showinfo("Liveness Check", "Please follow the instruction on the screen.")

    frame_count = 0
    known_face_locations = []
    known_face_names = []

    while True:
        ret, frame = vs.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not liveness_confirmed:
            cv2.putText(frame, challenge_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            boxes = face_recognition.face_locations(rgb_frame, model="hog")
            if boxes:
                current_face_center_x = (boxes[0][3] + boxes[0][1]) // 2
                if initial_face_x is None: initial_face_x = current_face_center_x
                movement = current_face_center_x - initial_face_x
                
                if (challenge_direction == "RIGHT" and movement > LIVENESS_MOVEMENT_THRESHOLD) or \
                   (challenge_direction == "LEFT" and movement < -LIVENESS_MOVEMENT_THRESHOLD):
                    liveness_confirmed = True
                    print("[INFO] Liveness Confirmed!")
            else:
                initial_face_x = None
        else:
            frame_count += 1
            if frame_count % VERIFY_PROCESS_EVERY_N_FRAMES == 0:
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
                known_face_locations = boxes
                known_face_names = names

            for ((top, right, bottom, left), name) in zip(known_face_locations, known_face_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow("Face Verification (Press 'q' to exit)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    vs.release()
    cv2.destroyAllWindows()
