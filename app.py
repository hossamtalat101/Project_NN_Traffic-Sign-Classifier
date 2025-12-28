import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# 1. إعدادات الصفحة والجمالية
st.set_page_config(page_title="كاشف إشارات المرور الذكي", layout="wide")

# تخصيص التصميم عبر CSS
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. تحميل الموديل وأسماء الإشارات
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    # تأكد من رفع ملف signnames.csv مع مشروعك
    labels = pd.read_csv('german-traffic-signs/signnames.csv')
    return model, labels

try:
    model, labels_df = load_assets()
except:
    st.error("خطأ: تأكد من وجود ملف الموديل (.h5) وملف الأسماء (.csv) في المجلد")

# 3. دالة المعالجة (نفس خطوات مشروعك بالضبط)
def process_img(img):
    img_array = np.array(img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized = equalized / 255.0
    resized = cv2.resize(normalized, (32, 32))
    return resized.reshape(1, 32, 32, 1), equalized

# 4. بناء الواجهة
st.title("🚦 نظام تصنيف إشارات المرور الألماني")
st.write("مشروع تخرج باستخدام الشبكات العصبية الالتفافية (CNN)")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 إدخال البيانات")
    uploaded_file = st.file_uploader("اختر صورة إشارة مرور...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة الأصلية", use_container_width=True)

with col2:
    st.subheader("🔍 نتائج التحليل")
    if uploaded_file:
        with st.spinner('جاري التحليل...'):
            # المعالجة والتوقع
            final_img, processed_view = process_img(image)
            prediction = model.predict(final_img)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            # جلب الاسم من ملف CSV
            sign_name = labels_df.loc[labels_df['ClassId'] == class_id, 'SignName'].values[0]

            # عرض النتيجة
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color: #28a745;'>التوقع: {sign_name}</h3>
                <p><strong>رقم الفئة:</strong> {class_id}</p>
                <p><strong>نسبة الثقة:</strong> {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # عرض شريط الثقة
            st.progress(int(confidence))
            
            # عرض ما يراه الموديل (لأغراض تعليمية)
            with st.expander("شاهد معالجة الموديل (Preprocessing)"):
                st.image(processed_view, caption="بعد التحويل لرمادي وتسوية التباين", width=150)
    else:
        st.info("الرجاء رفع صورة لبدء عملية التوقع.")
