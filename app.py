import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os
from gtts import gTTS
import base64

# 1. إعدادات الصفحة والتنسيق (يجب أن تكون في البداية)
st.set_page_config(page_title="نظام تصنيف الإشارات المرورية", layout="wide")

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    .stApp { background-color: #002b36; color: white; }
    .main-title { color: #d4af37; text-align: center; font-size: 2.5rem; font-weight: bold; padding: 10px; }
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
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. القاموس العربي
classes_ar = {
    0:'تحديد السرعة (20 كم/س)', 1:'تحديد السرعة (30 كم/س)', 2:'تحديد السرعة (50 كم/س)',
    3:'تحديد السرعة (60 كم/س)', 4:'تحديد السرعة (70 كم/س)', 5:'تحديد السرعة (80 كم/س)',
    6:'نهاية منطقة تحديد السرعة (80 كم/س)', 7:'تحديد السرعة (100 كم/س)', 8:'تحديد السرعة (120 كم/س)',
    9:'ممنوع التجاوز', 10:'ممنوع التجاوز للشاحنات', 11:'حق الأولوية عند التقاطع',
    12:'طريق ذو أولوية', 13:'أفسح الطريق (Yield)', 14:'قف (Stop)', 15:'ممنوع مرور المركبات',
    16:'ممنوع مرور الشاحنات', 17:'ممنوع الدخول', 18:'تحذير عام (خطر)', 19:'منحنى خطر لليسار',
    20:'منحنى خطر لليمين', 21:'منحنيات مزدوجة', 22:'طريق وعر (مطبات)', 23:'طريق زلق',
    24:'طريق يضيق من اليمين', 25:'أعمال طرق', 26:'إشارات ضوئية', 27:'عبور مشاة',
    28:'عبور أطفال', 29:'عبور دراجات هوائية', 30:'احذر من الجليد/الثلج',
    31:'عبور حيوانات برية', 32:'نهاية جميع القيود', 33:'إلزام بالاتجاه لليمين',
    34:'إلزام بالاتجاه لليسار', 35:'إلزام بالاتجاه للأمام فقط', 36:'إلزام للأمام أو اليمين',
    37:'إلزام للأمام أو اليسار', 38:'ابق على اليمين', 39:'ابق على اليسار',
    40:'دوار إلزامي', 41:'نهاية منع التجاوز', 42:'نهاية منع التجاوز للشاحنات'
}

# 3. الدوال البرمجية (Grad-CAM وتحميل الموديل)
def get_gradcam_heatmap(img_array, model):
    try:
        last_conv_layer_name = [layer.name for layer in model.layers if "conv2d" in layer.name][-1]
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return np.zeros((32,32))

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    return model

model = load_assets()

# 4. واجهة المستخدم
st.markdown('<h1 class="main-title"><i class="fas fa-crown"></i> المحلل الذكي الفاخر</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="custom-card"><i class="fas fa-image gold-icon"></i><h3>المدخلات</h3></div>', unsafe_allow_html=True)
    source = st.radio("اختر الوسيلة:", ["رفع ملف", "الكاميرا"], horizontal=True)
    if source == "رفع ملف":
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("")

with col2:
    if uploaded_file is not None:
        # المعالجة
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        equ = cv2.equalizeHist(gray)
        processed = cv2.resize(equ, (32, 32)).reshape(1, 32, 32, 1) / 255.0
        
        # التوقع
        preds = model.predict(processed)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        result_ar = classes_ar.get(idx, "إشارة غير معروفة")

        # عرض النتائج
        st.markdown(f"""
            <div class="custom-card">
                <i class="fas fa-check-double gold-icon"></i>
                <h2 style="color:#d4af37;">{result_ar}</h2>
                <p style="color:#3fb950;">نسبة الثقة: {confidence:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # الخريطة الحرارية والصوت
        heatmap = get_gradcam_heatmap(processed, model)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))), cv2.COLORMAP_JET)
        cam_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        tts = gTTS(text=f"انتبه، {result_ar}", lang='ar')
        tts.save('alert.mp3')
        st.audio('alert.mp3')

        t1, t2 = st.tabs(["الصورة الأصلية", "تحليل AI Focus"])
        t1.image(img, use_container_width=True)
        t2.image(cam_img, caption="Grad-CAM: المناطق التي ركز عليها النظام", use_container_width=True)
    else:
        st.info("يرجى رفع صورة للبدء بالتحليل.")

# 5. فريق التطوير
st.markdown("<br><hr style='border-color:#d4af37;'><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center;">
        <h3 style="color: #d4af37;"><i class="fas fa-users-gear"></i> فريق التطوير</h3>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div class="custom-card" style="min-width: 140px; padding: 10px; border-width: 1px;">Hossam</div>
            <div class="custom-card" style="min-width: 140px; padding: 10px; border-width: 1px;">Fatteh</div>
            <div class="custom-card" style="min-width: 140px; padding: 10px; border-width: 1px;">Osama</div>
        </div>
    </div>
""", unsafe_allow_html=True)
