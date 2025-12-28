import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# 1. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="Traffic Sign AI Explorer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. إضافة لمسات جمالية باستخدام CSS
st.markdown("""
    <style>
    /* تغيير خلفية التطبيق */
    .stApp {
        background-color: #0e1117;
    }
    /* تنسيق الحاويات (Cards) */
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
    }
    /* تنسيق النصوص */
    h1, h2, h3 {
        color: #58a6ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
        color: #238636;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. وظائف التحميل والمعالجة
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    labels = pd.read_csv('german-traffic-signs/signnames.csv')
    return model, labels

def process_image(img):
    img_array = np.array(img)
    # المعالجة الخاصة بك
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized = equalized / 255.0
    resized = cv2.resize(normalized, (32, 32))
    return resized.reshape(1, 32, 32, 1), equalized

# تحميل البيانات
model, labels_df = load_assets()

# 4. تصميم الهيكل (Layout)
# --- الشريط الجانبي ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2555/2555013.png", width=100)
    st.title("الإعدادات")
    st.info("هذا النظام يستخدم شبكة CNN مدربة على 43 فئة من إشارات المرور.")
    st.markdown("---")
    source = st.radio("اختر طريقة الإدخال:", ["رفع صورة", "الكاميرا الحية"])

# --- القسم الرئيسي ---
st.title("🚦 المحلل الذكي لإشارات المرور")
st.markdown("قم برفع صورة الإشارة ليقوم الذكاء الاصطناعي بتصنيفها وتحليلها فوراً.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 مدخلات الصورة")
    if source == "رفع صورة":
        input_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        input_file = st.camera_input("التقط صورة للإشارة")

    if input_file:
        image = Image.open(input_file)
        st.image(image, caption="الصورة الأصلية", use_container_width=True)

with col2:
    st.subheader("🎯 نتائج التحليل")
    if input_file:
        with st.spinner('جاري التحليل واستخراج الأنماط...'):
            final_img, processed_view = process_image(image)
            prediction = model.predict(final_img)
            class_id = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            sign_name = labels_df.loc[labels_df['ClassId'] == class_id, 'SignName'].values[0]

            # عرض النتيجة في حاوية مخصصة
            st.markdown(f"""
                <div class="metric-card">
                    <p style="color: #8b949e; margin-bottom: 5px;">الإشارة المتوقعة</p>
                    <p class="result-text">{sign_name}</p>
                    <hr style="border-color: #30363d;">
                    <p style="color: #8b949e;">درجة الثقة: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # عرض شريط التقدم الملون
            st.progress(int(confidence))
            
            # قسم "ماذا يرى الموديل؟"
            with st.expander("🛠️ عرض المعالجة الرقمية (X-Ray View)"):
                c1, c2 = st.columns(2)
                c1.image(processed_view, caption="بعد معالجة التباين", width=150)
                c2.write("هنا نقوم بتوحيد الإضاءة لضمان دقة التنبؤ في مختلف الظروف الجوية.")
    else:
        st.warning("في انتظار تزويدنا بصورة للبدء...")

# إضافة فوتر بسيط
st.markdown("---")
st.caption("مشروع تخرج - تطوير باستخدام Streamlit & TensorFlow")
