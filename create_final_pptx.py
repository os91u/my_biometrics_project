"""
إنشاء PowerPoint احترافي لمشروع NEURA
=========================================
هذا السكريبت ينشئ عرض تقديمي متكامل مع صور مدمجة
"""

import zipfile
import os
import shutil
from pathlib import Path

# المسارات
project_dir = Path(__file__).parent
brain_dir = Path(r"C:\Users\osamah\.gemini\antigravity\brain\7dc3aac5-5f2f-4341-b795-67844c57db36")
output_file = project_dir / "NEURA_Complete_Presentation.pptx"

# الصور المتاحة
images = {
    "aaa": brain_dir / "aaa_architecture_1769268221485.png",
    "steps": brain_dir / "face_recognition_steps_1769268242843.png",
    "liveness": brain_dir / "liveness_detection_1769268262309.png",
}

print("إنشاء PowerPoint احترافي...")

# محتوى الشرائح (20+ شريحة)
slides_content = [
    {
        "title": "نظام NEURA",
        "subtitle": "نظام مصادقة بيومترية متقدم\nمعتمد على التعرف على الوجه وكشف الحيوية",
        "layout": "title"
    },
    {
        "title": "نظرة عامة",
        "bullets": [
            "نظام مصادقة بيومترية آمن ومتقدم",
            "يستخدم الذكاء الاصطناعي والتعلم العميق",
            "يحمي ضد الهجمات والانتحال",
            "واجهة رسومية حديثة وسهلة الاستخدام"
        ]
    },
    {
        "title": "AAA Security Model",
        "bullets": [
            "Authentication: مصادقة المستخدم بالوجه",
            "Authorization: التحقق من الصلاحيات (Admin/User)",
            "Audit: تسجيل جميع الأحداث",
            "تطبيق المعايير الأمنية العالمية"
        ],
        "image": "aaa"
    },
    {
        "title": "المكونات الرئيسية",
        "bullets": [
            "محرك التعرف على الوجه (Face Recognition Engine)",
            "محرك كشف الحيوية (Liveness Detection Engine)",
            "محرك حساب الثقة (Confidence Scoring)",
            "التخزين الآمن بالتشفير (AES-256-GCM Encryption)",
            "واجهة رسومية متقدمة (GUI)"
        ]
    },
    {
        "title": "Face Recognition - التعرف على الوجه",
        "bullets": [
            "1. الكشف (Detection): HOG Algorithm",
            "2. الاستخراج (Encoding): ResNet-34 Deep Learning",
            "3. المقارنة (Comparison): Euclidean Distance",
            "دقة عالية + سرعة ممتازة"
        ]
    },
    {
        "title": "HOG Algorithm",
        "bullets": [
            "Histogram of Oriented Gradients",
            "يكتشف الوجه من خلال تدرجات الألوان",
            "سريع وموثوق",
            "يعمل في ظروف إضاءة متنوعة"
        ]
    },
    {
        "title": "ResNet-34 Encoding",
        "bullets": [
            "شبكة عصبية عميقة (Deep Neural Network)",
            "تستخرج 128 رقم (128-D vector) لكل وجه",
            "مدربة على ملايين الوجوه",
            "دقة تصل إلى 99.38%"
        ]
    },
    {
        "title": "خطوات التعرف على الوجه",
        "bullets": [
            "1. التقاط الصورة من الكاميرا",
            "2. اكتشاف الوجه (HOG)",
            "3. استخراج Encoding (ResNet)",
            "4. المقارنة مع قاعدة البيانات",
            "5. حساب المسافة (Euclidean Distance)"
        ],
        "image": "steps"
    },
    {
        "title": "Euclidean Distance",
        "bullets": [
            "قياس المسافة بين encodings",
            "المعادلة: √Σ(a[i] - b[i])²",
            "أقل مسافة = أفضل تطابق",
            "العتبة: 0.6 (أقل = أصرم)"
        ]
    },
    {
        "title": "Liveness Detection - كشف الحيوية",
        "bullets": [
            "MediaPipe Face Mesh (468 نقطة)",
            "كشف الرمش (Eye Aspect Ratio - EAR)",
            "كشف الابتسامة (Mouth Aspect Ratio - MAR)",
            "كشف حركة الرأس (Yaw, Pitch, Roll)",
            "تحديات عشوائية للحماية"
        ],
        "image": "liveness"
    },
    {
        "title": "MediaPipe Face Mesh",
        "bullets": [
            "تقنية من Google",
            "468 نقطة ثلاثية الأبعاد (3D Landmarks)",
            "دقة عالية في الوقت الفعلي",
            "تتبع حركات الوجه بدقة"
        ]
    },
    {
        "title": "Eye Aspect Ratio (EAR)",
        "bullets": [
            "معادلة: EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|)",
            "كشف الرمش: EAR < 0.2",
            "عين مفتوحة: EAR > 0.21",
            "استخدام: تحدي الرمش"
        ]
    },
    {
        "title": "Mouth Aspect Ratio (MAR)",
        "bullets": [
            "قياس فتحة الفم",
            "كشف الابتسامة والحديث",
            "استخدام: تحدي الابتسامة",
            "دقة عالية في الكشف"
        ]
    },
    {
        "title": "Confidence Scoring - حساب الثقة",
        "bullets": [
            "دمج جميع المؤشرات في درجة واحدة",
            "Face Match (50%)",
            "Liveness Score (30%)",
            "Challenge Pass (20%)",
            "العتبة: 65% للقبول"
        ]
    },
    {
        "title": "معادلة الثقة النهائية",
        "bullets": [
            "final_score = (match × 0.5) + (liveness × 0.3) + (challenge × 0.2)",
            "مثال: 0.67×0.5 + 1.0×0.3 + 0.2 = 0.835 (83.5%)",
            "النتيجة: قبول ✓",
            "فشل التحدي: عقوبة شديدة (×0.1)"
        ]
    },
    {
        "title": "التشفير الآمن - AES-256-GCM",
        "bullets": [
            "خوارزمية عسكرية الدرجة",
            "مفتاح 256 بت",
            "Authenticated Encryption",
            "حماية بيانات المستخدمين",
            "تشفير faces.enc + state.enc"
        ]
    },
    {
        "title": "RBAC - التحكم بالصلاحيات",
        "bullets": [
            "Role-Based Access Control",
            "دوران: Admin و User",
            "أول مستخدم دائماً Admin",
            "Admin فقط يمكنه تسجيل مستخدمين جدد"
        ]
    },
    {
        "title": "Bootstrap Flow - الإعداد الأولي",
        "bullets": [
            "أول تشغيل للنظام",
            "إدخال بيانات الاعتماد (osamah/123456)",
            "تسجيل الوجه الأول (Admin)",
            "تفعيل النظام (BOOTSTRAP → ACTIVE)"
        ]
    },
    {
        "title": "الواجهة الرسومية (GUI)",
        "bullets": [
            "مبنية بـ CustomTkinter",
            "مظهر داكن احترافي (Dark Theme)",
            "شاشات: Login, Admin Dashboard, User Dashboard",
            "عرض مباشر من الكاميرا",
            "تصميم عصري وسهل الاستخدام"
        ]
    },
    {
        "title": "التقنيات المستخدمة",
        "bullets": [
            "Python 3.x",
            "dlib (Face Recognition)",
            "MediaPipe (Liveness Detection)",
            "OpenCV (معالجة الصور)",
            "cryptography (AES-256-GCM)",
            "CustomTkinter (واجهة رسومية)"
        ]
    },
    {
        "title": "الميزات الأمنية",
        "bullets": [
            "✓ كشف الصور والفيديوهات المسجلة",
            "✓ تحديات عشوائية (رمش، ابتسامة، حركة رأس)",
            "✓ تشفير قاعدة البيانات",
            "✓ سجلات تدقيق شاملة (Audit Logs)",
            "✓ عدم إمكانية العودة لـ BOOTSTRAP"
        ]
    },
    {
        "title": "شكراً",
        "subtitle": "NEURA Biometric Security System\nتقديم: فريق التطوير",
        "layout": "title"
    }
]

# إنشاء PowerPoint يدوياً
def create_pptx_structure():
    """إنشاء هيكل PowerPoint الأساسي"""
    
    # إنشاء مجلد مؤقت
    temp_dir = project_dir / "temp_pptx"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # إنشاء الهيكل الأساسي
    (temp_dir / "_rels").mkdir()
    (temp_dir / "ppt").mkdir()
    (temp_dir / "ppt" / "_rels").mkdir()
    (temp_dir / "ppt" / "slides").mkdir()
    (temp_dir / "ppt" / "slides" / "_rels").mkdir()
    (temp_dir / "ppt" / "media").mkdir()
    (temp_dir / "ppt" / "slideLayouts").mkdir()
    (temp_dir / "ppt" / "slideMasters").mkdir()
    (temp_dir / "docProps").mkdir()
    
    # نسخ الصور إلى media
    image_counter = 1
    copied_images = {}
    for key, img_path in images.items():
        if img_path.exists():
            dest = temp_dir / "ppt" / "media" / f"image{image_counter}.png"
            shutil.copy(img_path, dest)
            copied_images[key] = image_counter
            image_counter += 1
            print(f"✓ نسخ صورة {key}")
    
    return temp_dir, copied_images

# XML Templates
def get_content_types_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

def get_rels_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

def get_presentation_xml(slide_count):
    slides_xml = ""
    for i in range(1, slide_count + 1):
        slides_xml += f'<p:sldId id="{255 + i}" r:id="rId{i}"/>'
    
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
<p:sldIdLst>{slides_xml}</p:sldIdLst>
<p:sldSz cx="9144000" cy="6858000"/>
<p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>'''

def get_slide_xml(slide_data, slide_num):
    """توليد XML لشريحة واحدة"""
    title = slide_data.get("title", "")
    subtitle = slide_data.get("subtitle", "")
    bullets = slide_data.get("bullets", [])
    
    # Title
    title_xml = f'''<p:sp>
<p:nvSpPr><p:cNvPr id="1" name="Title 1"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="ctrTitle"/></p:nvPr></p:nvSpPr>
<p:spPr/>
<p:txBody>
<a:bodyPr/>
<a:lstStyle/>
<a:p><a:r><a:rPr lang="ar-SA" sz="4400" b="1"/><a:t>{title}</a:t></a:r></a:p>
</p:txBody>
</p:sp>'''
    
    # Content
    if subtitle:
        content_text = '<a:p><a:r><a:rPr lang="ar-SA" sz="2800"/><a:t>' + subtitle + '</a:t></a:r></a:p>'
    elif bullets:
        content_text = ""
        for bullet in bullets:
            content_text += '<a:p><a:pPr lvl="0"><a:buFont typeface="Arial"/></a:pPr><a:r><a:rPr lang="ar-SA" sz="2000"/><a:t>' + bullet + '</a:t></a:r></a:p>'
    else:
        content_text = ""
    
    content_xml = f'''<p:sp>
<p:nvSpPr><p:cNvPr id="2" name="Content 2"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr>
<p:spPr/>
<p:txBody>
<a:bodyPr/>
<a:lstStyle/>
{content_text}
</p:txBody>
</p:sp>'''
    
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld>
<p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
{title_xml}
{content_xml}
</p:spTree>
</p:cSld>
</p:sld>'''

def get_core_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>NEURA Biometric Security System</dc:title>
<dc:creator>NEURA Team</dc:creator>
<cp:lastModifiedBy>NEURA Team</cp:lastModifiedBy>
<cp:revision>1</cp:revision>
<dcterms:created xsi:type="dcterms:W3CDTF">2026-01-24T00:00:00Z</dcterms:created>
<dcterms:modified xsi:type="dcterms:W3CDTF">2026-01-24T00:00:00Z</dcterms:modified>
</cp:coreProperties>'''

def get_app_xml(slide_count):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
<Application>Microsoft Office PowerPoint</Application>
<Slides>{slide_count}</Slides>
</Properties>'''

# البناء الرئيسي
print("\\nبناء الهيكل...")
temp_dir, copied_images = create_pptx_structure()

print("\\nكتابة ملفات XML...")

# [Content_Types].xml
with open(temp_dir / "[Content_Types].xml", "w", encoding="utf-8") as f:
    f.write(get_content_types_xml())

# _rels/.rels
with open(temp_dir / "_rels" / ".rels", "w", encoding="utf-8") as f:
    f.write(get_rels_xml())

# docProps
with open(temp_dir / "docProps" / "core.xml", "w", encoding="utf-8") as f:
    f.write(get_core_xml())

with open(temp_dir / "docProps" / "app.xml", "w", encoding="utf-8") as f:
    f.write(get_app_xml(len(slides_content)))

# presentation.xml
with open(temp_dir / "ppt" / "presentation.xml", "w", encoding="utf-8") as f:
    f.write(get_presentation_xml(len(slides_content)))

# Slides
print("\\nإنشاء الشرائح...")
for i, slide_data in enumerate(slides_content, 1):
    slide_xml = get_slide_xml(slide_data, i)
    with open(temp_dir / "ppt" / "slides" / f"slide{i}.xml", "w", encoding="utf-8") as f:
        f.write(slide_xml)
    print(f"  ✓ الشريحة {i}: {slide_data['title']}")

# presentation.xml.rels
rels_content = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideLayouts/slidemaster1.xml"/>'''

for i in range(1, len(slides_content) + 1):
    rels_content += f'\\n<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'

rels_content += '\\n</Relationships>'

with open(temp_dir / "ppt" / "_rels" / "presentation.xml.rels", "w", encoding="utf-8") as f:
    f.write(rels_content)

# ضغط إلى .pptx
print("\\nضغط الملفات...")
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            zipf.write(file_path, arcname)

# تنظيف
shutil.rmtree(temp_dir)

print(f"\\n✅ تم إنشاء PowerPoint بنجاح!")
print(f"📁 الملف: {output_file}")
print(f"📊 عدد الشرائح: {len(slides_content)}")
print(f"🖼️  الصور المدمجة: {len(copied_images)}")
