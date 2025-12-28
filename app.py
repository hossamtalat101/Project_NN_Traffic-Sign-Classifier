import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os

# أضف هذه المكتبات في أعلى ملف app.py
from gtts import gTTS
import base64

# 1. القاموس العربي الكامل الذي أرسلته (ضعه داخل الكود)
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

# 2. دالة Grad-CAM (المسؤولة عن الخريطة الحرارية)
def get_gradcam_heatmap(img_array, model):
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

# 3. تعديل جزء التوقع في app.py لإضافة الصوت والخريطة الحرارية
if file:
    # ... (كود المعالجة السابق) ...
    heatmap = get_gradcam_heatmap(final_input, model)
    
    # تحويل النص لصوت
    tts = gTTS(text=f"انتبه، {classes_ar[id]}", lang='ar')
    tts.save('alert.mp3')
    
    # عرض الصوت في Streamlit
    st.audio('alert.mp3', format='audio/mp3')
    
    # دمج الخريطة الحرارية مع الصورة المعالجة للعرض
    heatmap_resized = cv2.resize(heatmap, (32, 32))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # عرض النتائج
    st.markdown(f"### النتيجة بالعربية: {classes_ar[id]}")

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
