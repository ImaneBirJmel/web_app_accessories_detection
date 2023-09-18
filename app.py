from pathlib import Path
from PIL import Image
import streamlit as st
# Importez le module de configuration et les fonctions utilitaires
import config
from utils import load_model, infer_uploaded_image, infer_uploaded_webcam, seq_detection
import cv2
# Configuration de la mise en page de la page
st.set_page_config(
    page_title="Détection du Séquençage des Accessoires de Câbles Haute Tension Utilisant un Modèle DL",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Titre principal de la page
st.title("Détection du Séquençage des Accessoires de Câbles Haute Tension Utilisant un Modèle DL")
# Barre latérale
st.sidebar.header("Configuration du Modèle DL")
# Sélection de la confiance du modèle à partir de la barre latérale
confidence = float(st.sidebar.slider(
    "Sélectionnez la Confiance du Modèle", 30, 100, 50)) / 100
# Chemin du modèle pré-entraîné
model_path = "best.pt"
# Chargez le modèle DL pré-entraîné
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Impossible de charger le modèle. Veuillez vérifier le chemin spécifié : {model_path}")
# Sélectionnez la source (image ou webcam) depuis la barre latérale
source_selectbox = st.sidebar.selectbox(
    "Sélectionnez la Source",
    config.SOURCES_LIST
)
# Gestion de la source sélectionnée
source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    # Effectuez la détection sur une image téléchargée
    labels, image_path = infer_uploaded_image(confidence, model)
    # Appelez la fonction seq_detection avec les labels et le chemin de l'image pour détecter le #séquençage des objets
    seq_detection(labels, image_path)

elif source_selectbox == config.SOURCES_LIST[1]: # Webcam
    # Effectuez la détection en temps réel à partir de la webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Actuellement, seules les sources 'Image' et 'Webcam' sont prises en charge")
