from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_webcam, seq_detection

import cv2
# setting page layout
st.set_page_config(
    page_title="Detection of HV Cable Accessories Sequencing Using DL Model",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Detection of High-Voltage Cable Accessories Sequencing Using DL Model")

# sidebar
st.sidebar.header("DL Model Config")


confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = "best.pt"

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# 
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)
###


###
source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    labels, image_path=infer_uploaded_image(confidence, model)  # Obtenez les labels et le chemin de l'image 
    seq_detection(labels, image_path) # Appelez la fonction seq_detection avec les labels et le chemin de l'image pour detecter le séquencement des objets

elif source_selectbox == config.SOURCES_LIST[1]: # Webcam
    infer_uploaded_webcam(confidence, model)
    
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")
