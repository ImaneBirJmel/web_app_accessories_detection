from pathlib import Path 
import sys
# Obtient le chemin absolu du fichier courant
chemin_fichier = Path(__file__).resolve()
# Obtient le répertoire parent du fichier courant
chemin_racine = chemin_fichier.parent
# Ajoute le chemin racine à la liste sys.path s'il n'est pas déjà présent
if chemin_racine not in sys.path:
    sys.path.append(str(chemin_racine))
# Obtient le chemin relatif du répertoire racine par rapport au répertoire de travail courant
RACINE = chemin_racine.relative_to(Path.cwd())
# Liste des sources
LISTE_SOURCES = ["Image", "Webcam"]
