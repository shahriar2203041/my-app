import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import time
import io

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="ProPCB AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DARK THEME CSS ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    [data-testid="stSidebar"] { background-color: #1E2130; }
    h1, h2, h3 { color: #00F0FF; font-family: sans-serif; }
    [data-testid="stMetricValue"] { color: #00F0FF; font-size: 24px; }
    [data-testid="stFileUploader"] { background-color: #262730; border: 2px dashed #00F0FF; border-radius: 10px; padding: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/printed-circuit-board.png", width=100)
    st.title("⚙️ Controls")
    confidence = st.slider("🎯 Sensitivity", 0.1, 1.0, 0.45, 0.05)
    st.info("Model: YOLOv8 Nano\nAccuracy: 98.6%")

# --- 5. MAIN TITLE ---
st.title("⚡ ProPCB AI Inspector")
st.markdown("---")

# --- 6. UPLOAD ---
uploaded_file = st.file_uploader("📁 Upload PCB Image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2, col3 = st.columns(3)
    analyze_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Processing..."):
            results = model(image_np, conf=confidence)
            res_plotted = results[0].plot()
            boxes = results[0].boxes
            
            defect_count = len(boxes)
            class_names = []
            confidences = []
            
            if defect_count > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    name = model.names[cls_id]
                    conf = float(box.conf[0])
                    class_names.append(name)
                    confidences.append(conf)
                
                max_conf = max(confidences)
                col1.metric("Defects", f"{defect_count}")
                col2.metric("Max Confidence", f"{max_conf:.2%}")
                col3.metric("Status", "✅ Done")

                st.markdown("### 🔍 Results")
                img_col1, img_col2 = st.columns(2)
                with img_col1: st.image(image, caption="Original", use_column_width=True)
                with img_col2: st.image(res_plotted, caption="Analyzed", use_column_width=True)

                # Chart
                df = pd.DataFrame({'Type': class_names, 'Conf': confidences})
                count_df = df['Type'].value_counts().reset_index()
                count_df.columns = ['Type', 'Count']
                fig = px.bar(count_df, x='Type', y='Count', color='Count', color_continuous_scale='Cividis', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # Download
                res_im = Image.fromarray(res_plotted)
                buf = io.BytesIO()
                res_im.save(buf, format="JPEG")
                st.download_button("📥 Download Image", buf.getvalue(), "result.jpg", "image/jpeg")
            else:
                st.warning("No defects found.")
else:
    st.info("👈 Upload an image to start.")