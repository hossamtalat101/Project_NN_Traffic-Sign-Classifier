import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام تصنيف الإشارات المرورية", layout="wide")

# 2. التنسيق الجمالي (Teal & Gold)
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    .stApp { background-color: #002b36; color: white; }
    .main-title { color: #d4af37; text-align: center; font-size: 3rem; font-weight: bold; padding: 20px; }
    .custom-card { 
        background-color: #073642; 
        border: 2px solid #d4af37; 
        padding: 20px; 
        border-radius: 15px; 
        color: white; 
        text-align: center;
        margin-bottom: 20px;
    }
    .gold-icon { color: #d4af37; font-size: 2rem; margin-bottom: 10px; }
    .stProgress > div > div > div > div { background-color: #d4af37; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. تحميل الموديل والبيانات
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    csv_path = 'german-traffic-signs/signnames.csv'
    if os.path.exists(csv_path):
        labels = pd.read_csv(csv_path)
    else:
        labels = pd.read_csv('signnames.csv')
    return model, labels

model, labels_df = load_assets()

# 4. الواجهة الرئيسية
st.markdown('<h1 class="main-title"><i class="fas fa-traffic-light"></i> نظام التحليل الذكي الفاخر</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="custom-card"><i class="fas fa-camera-retro gold-icon"></i><h3>منطقة الإدخال</h3></div>', unsafe_allow_html=True)
    source = st.radio("اختر طريقة الإدخال:", ["رفع صورة", "الكاميرا الحية"], horizontal=True)
    
    if source == "رفع صورة":
        file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        file = st.camera_input("")

with col2:
    if file:
        img = Image.open(file)
        # المعالجة
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(gray)
        final = cv2.resize(equalized, (32, 32)).reshape(1, 32, 32, 1) / 255.0
        
        # التوقع
        res = model.predict(final)
        id = np.argmax(res)
        conf = np.max(res) * 100
        name = labels_df.loc[labels_df['ClassId'] == id, 'SignName'].values[0]
        
        st.markdown(f"""
            <div class="custom-card">
                <i class="fas fa-microchip gold-icon"></i>
                <h2 style="color:#d4af37;">النتيجة المتوقعة</h2>
                <p style="font-size:1.8rem;">{name}</p>
                <p style="color:#3fb950;">دقة التنبؤ: {conf:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)
        st.progress(int(conf))
        st.image(img, width=300)
    else:
        st.markdown('<div class="custom-card" style="border-style:dashed;"><i class="fas fa-hourglass-half gold-icon"></i><p>بانتظار الصورة...</p></div>', unsafe_allow_html=True)

# 5. فريق التطوير (أسفل الصفحة)
st.markdown("<br><hr style='border-color:#d4af37;'><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center;">
        <h3 style="color: #d4af37;"><i class="fas fa-users"></i> فريق التطوير</h3>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div class="custom-card" style="min-width: 150px; border-width: 1px;">
                <i class="fas fa-user-graduate gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin:0;">Hossam</p>
            </div>
            <div class="custom-card" style="min-width: 150px; border-width: 1px;">
                <i class="fas fa-user-graduate gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin:0;">Fatteh</p>
            </div>
            <div class="custom-card" style="min-width: 150px; border-width: 1px;">
                <i class="fas fa-user-graduate gold-icon" style="font-size: 1.2rem;"></i>
                <p style="margin:0;">Osama</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)
