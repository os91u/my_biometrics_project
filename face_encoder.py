# نستورد المكتبات اللازمة
import face_recognition
import pickle
import os
import cv2

# المسار إلى مجلد الصور المعروفة
KNOWN_FACES_DIR = "known_faces"
# اسم الملف الذي سنخزن فيه بصمات الوجوه
ENCODINGS_FILE = "encodings.pickle"

print("[INFO] Starting to process faces...")
# قائمة لتخزين بصمات الوجوه
known_encodings = []
# قائمة لتخزين أسماء الأشخاص
known_names = []

# نمر على كل الصور في مجلد الوجوه المعروفة
for filename in os.listdir(KNOWN_FACES_DIR):
    # نحصل على اسم الشخص من اسم الملف (بدون الامتداد)
    name = os.path.splitext(filename)[0]

    # المسار الكامل للصورة
    image_path = os.path.join(KNOWN_FACES_DIR, filename)

    print(f"[INFO] Processing image: {filename} for {name}")

    # نقرأ الصورة باستخدام OpenCV
    image = cv2.imread(image_path)
    # نحول الصورة من BGR (الذي يستخدمه OpenCV) إلى RGB (الذي تستخدمه face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # نحدد مواقع الوجوه في الصورة
    boxes = face_recognition.face_locations(rgb_image, model="hog")

    # نقوم بإنشاء بصمة الوجه
    # نفترض وجود وجه واحد فقط في كل صورة
    encodings = face_recognition.face_encodings(rgb_image, boxes)

    # نضيف البصمة والاسم إلى قوائمنا
    for encoding in encodings:
        known_encodings.append(encoding)
        known_names.append(name)

# نجهز البيانات لحفظها
data = {"encodings": known_encodings, "names": known_names}

# نفتح الملف ونقوم بحفظ البيانات باستخدام pickle
with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print(f"[INFO] Finished processing. {len(known_names)} faces encoded and saved to {ENCODINGS_FILE}")
