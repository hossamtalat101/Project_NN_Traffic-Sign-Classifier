import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from gtts import gTTS
import time
import os
import base64

# --- 1. إعدادات الصفحة والتنسيق (Modern UI Design) ---
st.set_page_config(page_title="SafeDrive AI System", layout="wide", page_icon="🚦")

def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
        * { font-family: 'Cairo', sans-serif; }
        .stApp { background: linear-gradient(135deg, #001524 0%, #002b36 100%); color: white; }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 20px;
        }
        
        .main-title { 
            background: linear-gradient(90deg, #d4af37, #f9d71c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center; font-size: 3.2rem; font-weight: 800; 
        }
        
        .result-box {
            text-align: center;
            border-radius: 15px;
            padding: 20px;
            border-top: 5px solid #d4af37;
            background: rgba(0, 0, 0, 0.3);
        }
        
        .status-pulse {
            width: 12px; height: 12px; background: #3fb950;
            border-radius: 50%; display: inline-block;
            box-shadow: 0 0 0 0 rgba(63, 185, 80, 1);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(63, 185, 80, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(63, 185, 80, 0); }
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# --- 2. القاموس العربي الكامل ---
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

# --- 3. الدوال التقنية (Processing & AI) ---
@st.cache_resource
def load_assets():
    # تأكد أن ملف الموديل موجود في نفس المجلد
    return tf.keras.models.load_model('traffic_sign_model.h5')

model = load_assets()

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
        return np.zeros((32, 32))

# --- 4. الهيكل الرئيسي للواجهة ---
st.markdown('<h1 class="main-title">SAFE DRIVE AI SYSTEM</h1>', unsafe_allow_html=True)

if 'history' not in st.session_state:
    st.session_state.history = []

# الشريط الجانبي
with st.sidebar:
    st.markdown("### <div class='status-pulse'></div> System: Active", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🕒 Recent Logs")
    for entry in reversed(st.session_state.history[-5:]):
        st.caption(f"✅ {entry['time']} - {entry['label']}")

tab1, tab2, tab3 = st.tabs(["🚀 Real-time Discovery", "🔬 AI Insights", "👥 Developers"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        source = st.radio("Input Method:", ["Upload Image", "Live Camera"], horizontal=True)
        if source == "Upload Image":
            uploaded_file = st.file_uploader("Select image...", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Snapshot")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file:
            start_time = time.time()
            img = Image.open(uploaded_file).convert('RGB')
            img_np = np.array(img)
            
            # المعالجة المسبقة
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            equ = cv2.equalizeHist(gray)
            processed = cv2.resize(equ, (32, 32)).reshape(1, 32, 32, 1) / 255.0
            
            # التوقع
            preds = model.predict(processed)
            idx = np.argmax(preds)
            confidence = np.max(preds) * 100
            result_ar = classes_ar.get(idx, "Unknown")
            
            # التسجيل في الذاكرة
            st.session_state.history.append({"time": time.strftime("%H:%M"), "label": result_ar})
            
            # عرض النتيجة
            st.markdown(f"""
                <div class="result-box">
                    <h2 style="color:#d4af37;">{result_ar}</h2>
                    <p>Accuracy Score: {confidence:.1f}%</p>
                    <p style="font-size:0.8rem; color:#839496;">Processed in: {(time.time()-start_time)*1000:.0f}ms</p>
                </div>
            """, unsafe_allow_html=True)
            
            # تنبيه صوتي
            tts = gTTS(text=f"انتبه، {result_ar}", lang='ar')
            tts.save('alert.mp3')
            st.audio('alert.mp3', format="audio/mp3", autoplay=True)
        else:
            st.info("System is ready. Please provide an input image.")

with tab2:
    if uploaded_file:
        heatmap = get_gradcam_heatmap(processed, model)
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlayed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        c1, c2 = st.columns(2)
        c1.image(img, caption="Original Input", use_container_width=True)
        c2.image(overlayed_img, caption="Grad-CAM: Heatmap (Focus Area)", use_container_width=True)
        
        st.markdown("#### Top 3 Probabilities:")
        top_3 = np.argsort(preds[0])[-3:][::-1]
        for i in top_3:
            st.write(f"**{classes_ar[i]}**: {preds[0][i]*100:.1f}%")
            st.progress(float(preds[0][i]))

with tab3:
    st.markdown("<div style='display: flex; justify-content: space-around;'>", unsafe_allow_html=True)
    for name in ["Hossam Talat", "Fatteh", "Osama"]:
        st.markdown(f"<div class='glass-card' style='text-align:center;'><h4>{name}</h4><p>Engineer</p></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
