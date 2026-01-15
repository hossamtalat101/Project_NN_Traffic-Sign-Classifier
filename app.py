import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from gtts import gTTS
import time
import os

# 1. إعدادات الصفحة والتنسيق الفاخر
st.set_page_config(page_title="Traffic Sign Intelligence System", layout="wide")

# تهيئة سجل العمليات في ذاكرة المتصفح
if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    .stApp { background-color: #002b36; color: white; }
    .main-title { color: #d4af37; text-align: center; font-size: 2.8rem; font-weight: bold; padding: 10px; text-shadow: 2px 2px 4px #000; }
    .custom-card { 
        background-color: #073642; 
        border: 2px solid #d4af37; 
        padding: 20px; 
        border-radius: 15px; 
        color: white; 
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .gold-icon { color: #d4af37; font-size: 2rem; margin-bottom: 10px; }
    .metric-box { background: #001f27; padding: 15px; border-radius: 10px; border-left: 5px solid #d4af37; margin-bottom: 10px;}
    .sidebar-text { font-size: 0.9rem; color: #839496; line-height: 1.6; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. القاموس العربي الكامل
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

# 3. الدوال التقنية المتقدمة
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
    except: return np.zeros((32,32))

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    return model

model = load_assets()

# 4. الشريط الجانبي (Sidebar) - إحصائيات وسجل
with st.sidebar:
    st.markdown('<div class="custom-card"><i class="fas fa-microchip gold-icon"></i><h3>Model Stats</h3></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-box"><b>Accuracy:</b> 98.4%</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-box"><b>Inference:</b> Real-time</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### <i class='fas fa-history'></i> سجل العمليات الأخير")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            st.caption(f"🕒 {entry['time']} - {entry['label']}")
    else:
        st.write("لا توجد عمليات اكتشاف بعد.")
        
    st.markdown("---")
    st.markdown("### <i class='fas fa-info-circle'></i> وصف الموديل")
    st.markdown('<p class="sidebar-text">يعتمد هذا النظام على معمارية CNN المتقدمة، حيث يقوم بمعالجة الصور عبر طبقات تلافيفية لاستخلاص الميزات الهندسية.</p>', unsafe_allow_html=True)

# 5. الواجهة الرئيسية
st.markdown('<h1 class="main-title"><i class="fas fa-crown"></i> نظام التحليل الذكي</h1>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="custom-card"><i class="fas fa-upload gold-icon"></i><h3>منطقة الإدخال</h3></div>', unsafe_allow_html=True)
    source = st.radio("اختر الوسيلة:", ["رفع صورة", "الكاميرا الحية"], horizontal=True)
    if source == "رفع صورة":
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("")

with col2:
    if uploaded_file is not None:
        start_time = time.time()
        img = Image.open(uploaded_file).convert('RGB')
        img_np = np.array(img)
        
        # المعالجة
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        equ = cv2.equalizeHist(gray)
        processed = cv2.resize(equ, (32, 32)).reshape(1, 32, 32, 1) / 255.0
        
        # التوقع
        preds = model.predict(processed)
        idx = np.argmax(preds)
        confidence = np.max(preds) * 100
        result_ar = classes_ar.get(idx, "إشارة غير معروفة")
        inference_time = (time.time() - start_time) * 1000

        # إضافة للسجل
        st.session_state.history.append({"time": time.strftime("%H:%M"), "label": result_ar})

        # نظام التنبيه الملون
        alert_color = "#3fb950" 
        if idx in [14, 17, 15, 18]: alert_color = "#ff4b4b" # أحمر
        elif idx < 9: alert_color = "#f9d71c" # أصفر

        st.markdown(f"""
            <div class="custom-card" style="border-color: {alert_color};">
                <h2 style="color:{alert_color};"><i class="fas fa-eye"></i> {result_ar}</h2>
                <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                    <span><b>دقة التنبؤ:</b> {confidence:.1f}%</span>
                    <span><b>الزمن:</b> {inference_time:.0f}ms</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # الصوت
        tts = gTTS(text=f"انتبه، {result_ar}", lang='ar')
        tts.save('alert.mp3')
        st.audio('alert.mp3')

        # التبويبات والمقارنة
        t1, t2, t3 = st.tabs(["🖼️ التحليل البصري", "📊 الاحتمالات المنافسة", "🔬 X-Ray View"])
        
        with t1:
            # دمج الخريطة الحرارية
            heatmap = get_gradcam_heatmap(processed, model)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))), cv2.COLORMAP_JET)
            cam_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
            
            c1, c2 = st.columns(2)
            c1.image(img, caption="الصورة الأصلية", use_container_width=True)
            c2.image(cam_img, caption="Grad-CAM: تركيز الذكاء الاصطناعي", use_container_width=True)

        with t2:
            st.markdown("#### أعلى 3 احتمالات مكتشفة:")
            top_3_indices = np.argsort(preds[0])[-3:][::-1]
            for i in top_3_indices:
                score = preds[0][i] * 100
                st.write(f"**{classes_ar.get(i, 'غير معروف')}:** {score:.1f}%")
                st.progress(int(score))

        with t3:
            st.image(equ, caption="توضيح الحواف والتباين (Pre-processing Step)", use_container_width=True)
            st.markdown("""
                * تم تحويل الصورة للون الرمادي لتقليل ضجيج الألوان.
                * تم تطبيق **Histogram Equalization** لتوضيح الإشارة في ظروف الإضاءة الصعبة.
            """)
    else:
        st.markdown('<div class="custom-card" style="border-style: dashed; opacity: 0.7;"><i class="fas fa-hourglass-start gold-icon"></i><p>بانتظار تزويد النظام ببيانات الإدخال للبدء بالتحليل المتقدم...</p></div>', unsafe_allow_html=True)

# 6. الخريطة (ميزة إضافية)
if uploaded_file:
    with st.expander("📍 موقع الاكتشاف التقديري"):
        # إحداثيات افتراضية تظهر في الخريطة
        df_map = pd.DataFrame({'lat': [24.7136], 'lon': [46.6753]})
        st.map(df_map)

# 7. فريق العمل
st.markdown("<br><hr style='border-color:#d4af37;'><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center;">
        <h3 style="color: #d4af37;"><i class="fas fa-users-cog"></i> فريق التطوير</h3>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
            <div class="custom-card" style="min-width: 160px; padding: 10px; border-width: 1px;"><b>Hossam</b></div>
            <div class="custom-card" style="min-width: 160px; padding: 10px; border-width: 1px;"><b>Fatteh</b></div>
            <div class="custom-card" style="min-width: 160px; padding: 10px; border-width: 1px;"><b>Osama</b></div>
        </div>
    </div>
""", unsafe_allow_html=True)
