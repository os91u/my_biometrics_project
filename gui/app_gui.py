# gui/app_gui.py

import tkinter as tk
from tkinter import font

class AppGUI:
    def __init__(self, root, add_face_callback, verify_face_callback):
        self.root = root
        self.root.title("Face Recognition System")
        
        # حجم النافذة وموقعها في وسط الشاشة
        window_width = 400
        window_height = 300
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        self.root.configure(bg='#2c3e50') # لون خلفية أنيق

        # إعدادات الخطوط
        title_font = font.Font(family="Helvetica", size=18, weight="bold")
        button_font = font.Font(family="Arial", size=12)

        # العنوان
        title_label = tk.Label(root, text="Main Menu", font=title_font, bg='#2c3e50', fg='white')
        title_label.pack(pady=20)

        # إطار للأزرار
        button_frame = tk.Frame(root, bg='#2c3e50')
        button_frame.pack(pady=10, padx=20, fill='x')

        # الأزرار
        add_button = tk.Button(button_frame, text="Add New Face", command=add_face_callback, font=button_font, bg='#1abc9c', fg='white', relief='flat', padx=10, pady=5)
        add_button.pack(fill='x', pady=5)

        verify_button = tk.Button(button_frame, text="Verify Face", command=verify_face_callback, font=button_font, bg='#3498db', fg='white', relief='flat', padx=10, pady=5)
        verify_button.pack(fill='x', pady=5)

        exit_button = tk.Button(button_frame, text="Exit", command=root.quit, font=button_font, bg='#e74c3c', fg='white', relief='flat', padx=10, pady=5)
        exit_button.pack(fill='x', pady=5)
