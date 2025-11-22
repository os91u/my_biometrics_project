# نستورد المكتبات اللازمة
import face_recognition
import pickle
import cv2

# اسم ملف البصمات
ENCODINGS_FILE = "encodings.pickle"
# يمكن تغيير الرقم إلى 1 أو 2 إذا كان لديك أكثر من كاميرا
CAMERA_SOURCE = 0

print("[INFO] Loading encodings...")
# نقوم بتحميل بصمات الوجوه المخزنة
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

# نفتح كاميرا الويب
print("[INFO] Starting video stream...")
vs = cv2.VideoCapture(CAMERA_SOURCE)

# حلقة مستمرة لقراءة الإطارات من الكاميرا
while True:
    # نقرأ الإطار الحالي من الكاميرا
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # نحدد مواقع الوجوه في الإطار الحالي
    boxes = face_recognition.face_locations(frame, model="hog")
    # نقوم بإنشاء بصمات للوجوه التي تم العثور عليها
    encodings = face_recognition.face_encodings(frame, boxes)

    names = []

    # نمر على كل بصمة وجه تم العثور عليها في الإطار
    for encoding in encodings:
        # نحاول مطابقة الوجه مع البصمات المخزنة
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown" # القيمة الافتراضية إذا لم يتم العثور على تطابق

        # إذا تم العثور على تطابق
        if True in matches:
            # نجد مواقع كل التطابقات
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # نحصي عدد مرات تطابق كل اسم
            for i in matched_idxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # نختار الاسم الأكثر تكراراً (الأكثر تطابقاً)
            name = max(counts, key=counts.get)

        names.append(name)

    # نمر على الوجوه التي تم التعرف عليها ومواقعها لرسمها على الشاشة
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # نرسم مستطيلاً حول الوجه
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # نكتب اسم الشخص تحت المستطيل
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # نعرض الفيديو على الشاشة
    cv2.imshow("Face Recognition", frame)

    # ننتظر الضغط على مفتاح 'q' للخروج
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# ننظف ونغلق كل شيء
print("[INFO] Cleaning up...")
cv2.destroyAllWindows()
vs.release()
