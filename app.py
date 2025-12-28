import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# 1. إعدادات الصفحة
st.set_page_config(
    page_title="Traffic Sign AI - Gold Edition",
    page_icon="🚦",
    layout="wide"
)

# 2. إضافة CSS مخصص للألوان البترولية والذهبية والأيقونات
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    /* الخلفية العامة */
    .stApp {
        background-color: #002b36; /* بترولي غامق جداً */
    }
    
    /* العنوان الرئيسي */
    .main-title {
        color: #d4af37; /* ذهبي */
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 30px;
    }

    /* الحاويات (Cards) */
    .custom-card {
        background-color: #073642; /* بترولي فاتح */
        border: 2px solid #d4af37; /* إطار ذهبي */
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        color: white;
        margin-bottom: 20px;
    }

    /* الأيقونات */
    .gold-icon {
        color: #d4af37;
        font-size: 2rem;
        margin-bottom: 10px;
    }

    /* التنسيق الجانبي */
    [data-testid="stSidebar"] {
        background-color: #001f27;
        border-right: 2px solid #d4af37;
    }
    
    .stProgress > div > div > div > div {
        background-color: #d4af37; /* شريط التقدم ذهبي */
    }
    </style>
    """, unsafe_allow_html=True)

# 3. وظائف التحميل (مع تصحيح المسار الخاص بك)
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    # التأكد من المسار حسب صورك السابقة
    csv_path = 'german-traffic-signs/signnames.csv'
    labels = pd.read_csv(csv_path)
    return model, labels

model, labels_df = load_assets()

# 4. واجهة المستخدم
st.markdown('<h1 class="main-title"><i class="fas fa-microchip"></i> نظام التحليل الذكي الفاخر</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="custom-card">
            <i class="fas fa-cloud-upload-alt gold-icon"></i>
            <h3>منطقة الإدخال</h3>
            <p style='color: #839496;'>قم برفع الصورة أو التقاطها لبدء المعالجة</p>
        </div>
    """, unsafe_allow_html=True)
    
    input_type = st.radio("اختر الوسيلة:", ["<i class='fas fa-file-image'></i> رفع ملف", "<i class='fas fa-camera'></i> كاميرا"], index=0, horizontal=True)
    
    if "رفع" in input_type:
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("")

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # معالجة الصورة
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        processed_img = cv2.resize(equalized, (32, 32)) / 255.0
        final_input = processed_img.reshape(1, 32, 32, 1)
        
        # التوقع
        preds = model.predict(final_input)
        class_id = np.argmax(preds)
        conf = np.max(preds) * 100
        name = labels_df.loc[labels_df['ClassId'] == class_id, 'SignName'].values[0]

        # عرض النتيجة بشكل فاخر
        st.markdown(f"""
            <div class="custom-card">
                <i class="fas fa-check-circle gold-icon"></i>
                <h2 style='color: #d4af37; margin:0;'>تم التعرف بنجاح</h2>
                <hr style='border-color: #d4af37;'>
                <p style='font-size: 1.5rem;'><b>إشارة:</b> {name}</p>
                <p style='font-size: 1.2rem; color: #3fb950;'><b>درجة الثقة:</b> {conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))
        st.image(image, caption="الصورة التي تم تحليلها", width=300)
    else:
        st.markdown("""
            <div class="custom-card" style='text-align: center; border-style: dashed;'>
                <i class="fas fa-hourglass-start gold-icon" style='animation: spin 2s linear infinite;'></i>
                <p>في انتظار بيانات الدخل...</p>
            </div>
            <style>
            @keyframes spin { 100% { transform:rotate(360deg); } }
            </style>
        """, unsafe_allow_html=True)

# شريط جانبي معلوماتي
with st.sidebar:
    st.markdown("<h2 style='color: #d4af37;'><i class='fas fa-info-circle'></i> معلومات النظام</h2>", unsafe_allow_html=True)
    st.write("نظام مدعوم بالذكاء الاصطناعي لتصنيف 43 نوعاً من إشارات المرور بدقة عالية.")
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h3 style="color: #d4af37; font-family: 'Segoe UI';">
            <i class="fas fa-users-cog"></i> فريق التطوير
        </h3>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 15px;">
            <div class="custom-card" style="padding: 15px 30px; border-width: 1px;">
                <i class="fas fa-user-tie gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin: 0; font-weight: bold;">Hossam</p>
            </div>
            <div class="custom-card" style="padding: 15px 30px; border-width: 1px;">
                <i class="fas fa-user-tie gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin: 0; font-weight: bold;">Fatteh</p>
            </div>
            <div class="custom-card" style="padding: 15px 30px; border-width: 1px;">
                <i class="fas fa-user-tie gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin: 0; font-weight: bold;">Osama</p>
            </div>
        </div>
        <p style="color: #839496; margin-top: 20px; font-size: 0.9rem;">
            تم التطوير بكل فخر باستخدام تقنيات Deep Learning & Computer Vision © 2025
        </p>
    </div>
""", unsafe_allow_html=True)
