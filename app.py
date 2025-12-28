import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import os
from gtts import gTTS
import base64

# 1. إعدادات الصفحة والتنسيق (Teal & Gold)
st.set_page_config(page_title="نظام تصنيف الإشارات المرورية", layout="wide")

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

# 2. القاموس العربي الكامل لـ 43 فئة
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

# 3. دالة Grad-CAM لرؤية تركيز الموديل
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

# 4. تحميل الموديل والبيانات
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('traffic_sign_model.h5')
    csv_path = 'german-traffic-signs/signnames.csv'
    if os.path.exists(csv_path):
        labels = pd.read_csv(csv_path)
    else:
        labels = pd.read_csv('signnames.csv')
    return model, labels
