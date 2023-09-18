from pathlib import Path
import sys
#Obtenez le chemin absolu du fichier courant
file_path = Path(__file__).resolve()
#Obtenir le répertoire parent du fichier actuel
root_path = file_path.parent
#Ajoutez le chemin racine à la liste sys.path s'il n'est pas déjà présent
if root_path not in sys.path:
    sys.path.append(str(root_path))
# Obtenir le chemin relatif du répertoire racine par rapport au répertoire de travail courant
ROOT = root_path.relative_to(Path.cwd())
# Source
SOURCES_LIST = ["Image", "Webcam"]
