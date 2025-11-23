import face_recognition
import pickle
import cv2
from scipy.spatial import distance as dist
import os

# --- إعدادات ---
ENCODINGS_FILE = "encodings.pickle"
CAMERA_SOURCE = 0
EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 2

# --- دوال مساعدة ---

def eye_aspect_ratio(eye):
    """يحسب نسبة أبعاد العين لتحديد الرمش."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def load_encodings():
    """يقوم بتحميل بصمات الوجوه من الملف، أو إنشاء ملف جديد إذا لم يكن موجوداً."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"encodings": [], "names": []}
    return data

def save_encodings(data):
    """يحفظ بصمات الوجوه في الملف."""
    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

# --- الوظائف الرئيسية ---

def add_new_face():
    """إضافة بصمة وجه جديدة باستخدام الكاميرا مع كشف الحيوية."""
    data = load_encodings()
    vs = cv2.VideoCapture(CAMERA_SOURCE)

    frame_counter = 0
    face_captured = False

    print("\n[INFO] Preparing to add new face...")
    print("[ACTION] Please look at the camera and BLINK your eyes.")

    while not face_captured:
        ret, frame = vs.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        is_blinking = False
        for face_landmarks in face_landmarks_list:
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_ASPECT_RATIO_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    is_blinking = True
                frame_counter = 0

        if is_blinking:
            print("[SUCCESS] Liveness confirmed (Blink detected)!")
            encodings = face_recognition.face_encodings(rgb_frame)
            if encodings:
                new_encoding = encodings[0]

                while True:
                    new_name = input("[INPUT] Face signature captured. Please enter the name for this person: ")
                    if new_name.strip(): # التأكد من أن الاسم ليس فارغاً
                        break
                    print("[ERROR] Name cannot be empty.")

                data["encodings"].append(new_encoding)
                data["names"].append(new_name)
                save_encodings(data)
                print(f"[SUCCESS] Face signature for '{new_name}' has been saved.")
                face_captured = True
            else:
                print("[WARNING] Blink detected, but could not capture face encoding. Please try again.")

        cv2.imshow("Add New Face - BLINK to capture (Press 'q' to cancel)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

def verify_face():
    """التحقق من بصمة الوجه مباشرة من الكاميرا."""
    data = load_encodings()
    if not data["encodings"]:
        print("\n[ERROR] No faces have been saved yet. Please add a face first.")
        return

    vs = cv2.VideoCapture(CAMERA_SOURCE)
    print("\n[INFO] Starting face verification...")
    print("[ACTION] Look at the camera. Recognition will start after a blink.")

    is_live = False
    while True:
        ret, frame = vs.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # كشف الحيوية أولاً
        if not is_live:
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
            for face_landmarks in face_landmarks_list:
                # (كود كشف الرمش مكرر هنا للتبسيط)
                left_ear = eye_aspect_ratio(face_landmarks['left_eye'])
                right_ear = eye_aspect_ratio(face_landmarks['right_eye'])
                ear = (left_ear + right_ear) / 2.0
                if ear < EYE_ASPECT_RATIO_THRESHOLD:
                    is_live = True # نعتبره حياً بمجرد أن يرمش
                    print("[SUCCESS] Liveness confirmed. Starting recognition...")

        # إذا تم تأكيد الحيوية، نبدأ بالتعرف
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vs.release()
    cv2.destroyAllWindows()

# --- القائمة الرئيسية للبرنامج ---
def main_menu():
    """يعرض القائمة الرئيسية ويتعامل مع اختيار المستخدم."""
    while True:
        print("\n" + "="*30)
        print("  Face Recognition System Menu")
        print("="*30)
        print("1. Add a new face signature")
        print("2. Verify a face")
        print("3. Exit")
        choice = input("Please enter your choice (1-3): ")

        if choice == '1':
            add_new_face()
        elif choice == '2':
            verify_face()
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
