import os
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import matplotlib.pyplot as plt
import numpy as np 
import sys
from pathlib import Path
# Importez le module de configuration pour accéder à la variable ROOT
import config
from config import ROOT  # Importez directement la variable ROOT

# Ajoutez le chemin racine à la liste sys.path s'il n'est pas déjà présent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Fonction pour afficher les images détectées
def _display_detected_frames(conf, model, st_frame, image):
    # Redimensionner l'image à une taille standard
    image = cv2.resize(image, (640, int(640 * (9 / 16))))
    # Prédire les objets dans l'image en utilisant le modèle YOLOv8 pré-entrainé
    res = model.predict(image, conf=conf)

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

# Fonction pour effectuer la détection sur une image téléchargée
def infer_uploaded_image(conf, model):
# Sélectionnez une image depuis la barre latérale
    source_img = st.sidebar.file_uploader(
        label="Choisir une image...",
        type=("jpg", "jpeg", "png", "bmp", "webp"),
        accept_multiple_files=False  # Permettre uniquement le téléchargement d'un seul fichier
    )

    # Affichez l'image téléchargée et les résultats de détection
col1, col2 = st.columns(2)
    labels = []
    image_path = ""
    with col1:
        if source_img is not None:
            # Créer un fichier temporaire pour stocker l'image téléchargée
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image_file:
                temp_image_file.write(source_img.read())
                image_path = temp_image_file.name
            uploaded_image = Image.open(image_path)
            # Ajout de l'image téléchargée à la page avec une légende
            st.image(
                image=uploaded_image,
                caption="Image téléchargée",
                use_column_width=True
            )
    if source_img:
        if st.button("Exécution"):
            with st.spinner("En cours d'exécution..."):
                # Convertir l'image téléchargée au format OpenCV (BGR)
                opencv_image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
                res = model.predict(opencv_image, conf=conf)
                boxes = res[0].boxes
                with col2:
                    # Afficher l'image détectée avec les boîtes englobantes
                    img_with_boxes = res[0].plot()[:, :, ::-1]
                    st.image(img_with_boxes, caption="Image détectée", use_column_width=True)
                    try:
                        with st.expander("Résultats de détection"):
                            names = boxes.cls.tolist()
                            cord = boxes.xywhn.tolist()
                            labels = []
                            for i in range(len(names)):
                                label = [names[i]]
                                label.extend(cord[i])
                                labels.append(label)
                            st.write(labels)
                          # Afficher le chemin d'image relatif
                            st.write(os.path.relpath(image_path, start=ROOT))  

                    except Exception as ex:
                        st.write("Aucune image n'est téléchargée !")
                        st.write(ex)
    return labels, image_path

# Fonction pour effectuer la détection en temps réel à partir de la webcam    
def infer_uploaded_webcam(conf, model):
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

# Fonction pour effectuer la séquence de détection et afficher les rectangles
def seq_detection(labels, image_path):
    image = cv2.imread(image_path)
    # Check if the image is loaded successfully
    if image is None:
        st.error(f"Failed to load the image at path: {image_path}")
    L = denormaliser(labels)
    L_sorted = sorted(L, key=lambda x: x[1])
    Y = [sublist[0] for sublist in L_sorted]    
    Cord = [[sublist[1], sublist[2], sublist[3], sublist[4]] for sublist in L_sorted]
    y1 = [0, 1, 2, 3, 4, 5]
    y2 = [5, 4, 3, 2, 1, 0]
    ##
    z1 = [1 if Y[i] == y1[i] else 0 for i in range(len(Y))]
    z2 = [1 if Y[i] == y2[i] else 0 for i in range(len(Y))]

    # Dessin des rectangles
    if sum(z1) == 6 or sum(z2) == 6:
        st.write("Les accessoires sont tous bien séquencés.")
        for i in range(6):
            draw_rectangle(image, Cord[i], (0, 255, 0))
    else:
        st.write("Il y a des accessoires qui sont mal séquencés.")
        if sum(z1) < sum(z2):
            for i in range(len(z2)):
                if z2[i] == 1:                   
                    draw_rectangle(image, Cord[i], (0, 255, 0))
                else:                    
                    draw_rectangle(image, Cord[i], (0, 0, 255))
        else:
            for j in range(len(z1)):
                if z1[j] == 1:                    
                    draw_rectangle(image, Cord[j], (0, 255, 0))
                else:                   
                    draw_rectangle(image, Cord[j], (0, 0, 255))
    # Convertir l'image de BGR à RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Afficher l'image avec les rectangles dans Streamlit
    st.image(image_rgb, use_column_width=True)

# Fonction pour dessiner des rectangles autour des objets détectés
def draw_rectangle(image, rect_coords, color):
    x, y, w, h = rect_coords
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness=3)

# Fonction pour dénormaliser les coordonnées des objets détectés
def denormaliser(L):
    L_denormaliser = [[sublist[0],int(sublist[1] * 640) - int(sublist[3] * 640 / 2), (int(sublist[2] * 640) - int(sublist[4] * 640 / 2)), int(sublist[3] * 640), int(sublist[4] * 640)] for sublist in L]
    return L_denormaliser
