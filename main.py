# main.py

import tkinter as tk
# نستورد الدوال من ملف منطق العمل
from face_logic import add_new_face_enhanced, verify_face

# نستورد كلاس الواجهة من ملف الواجهة
from gui.app_gui import AppGUI

def main():
    """
    نقطة انطلاق التطبيق.
    يقوم بإنشاء الواجهة الرسومية وربط الأزرار بالوظائف المنطقية.
    """
    # إنشاء النافذة الرئيسية لـ tkinter
    root = tk.Tk()
    
    # إنشاء نسخة من الواجهة الرسومية، وتمرير الدوال التي يجب أن تستدعيها الأزرار
    # هذا هو السطر الذي يربط الواجهة بالمنطق
    app = AppGUI(root, 
                 add_face_callback=add_new_face_enhanced, 
                 verify_face_callback=verify_face)
    
    # تشغيل الحلقة الرئيسية للتطبيق لعرض الواجهة
    root.mainloop()

# التأكد من تشغيل الدالة الرئيسية فقط عندما يتم تشغيل هذا الملف مباشرة
if __name__ == "__main__":
    main()
