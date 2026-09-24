import os
import matplotlib.pyplot as plt
from PIL import Image

# =====================================================================
# 2.1. Lister les classes et compter les images
# =====================================================================

TRAIN_DIR = "melanoma-cancer-dataset/train"

# os.listdir() retourne la liste des éléments dans un dossier
classes = sorted(os.listdir(TRAIN_DIR))
num_classes = len(classes)
print(f"Nombre de classes : {num_classes}")
print(f"Classes : {classes}\n")

# Compter les images par classe
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    print(f"{classe} : {nb_images} images")

# =====================================================================
# 2.2. Afficher des exemples
# =====================================================================

nb_images_par_classe = 6
nb_lignes = num_classes
nb_colonnes = nb_images_par_classe

# Création d'une grille de sous-graphiques
fig, axes = plt.subplots(nb_lignes, nb_colonnes, figsize=(14, 3 * num_classes))

# Transformation du tableau 2D d'axes en liste 1D pour itérer facilement
axes = axes.flatten()

index_axe = 0

for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    # Liste de tous les fichiers dans le dossier de la classe
    fichiers = sorted(os.listdir(chemin_classe))
    
    # Sélection des 2 premières images
    images_a_afficher = fichiers[:nb_images_par_classe]
    
    for fichier_img in images_a_afficher:
        # Construction propre du chemin vers l'image
        chemin_image = os.path.join(chemin_classe, fichier_img)
        
        # Chargement de l'image avec PIL
        img = Image.open(chemin_image)
        
        axes[index_axe].imshow(img)
        axes[index_axe].set_title(f"{classe}")
        axes[index_axe].axis("off")
        
        index_axe += 1

plt.suptitle("Exemples d'images du dataset Melanoma", fontsize=16)
# Ajustement automatique des marges pour éviter que les textes se chevauchent
plt.tight_layout()
plt.show()

# =====================================================================
# 2.3. Diagramme en barres de la distribution
# =====================================================================

classes = sorted(os.listdir(TRAIN_DIR))
num_classes = len(classes)


# Construction du dictionnaire counts
counts = {}
for classe in classes:
    chemin_classe = os.path.join(TRAIN_DIR, classe)
    nb_images = len(os.listdir(chemin_classe))
    # On ajoute l'entrée au dictionnaire : clé = nom de la classe, valeur = nombre
    counts[classe] = nb_images
    
print("Distribution des classes :", counts)

# Affichage le diagramme en barres
plt.figure(figsize=(8, 5)) 

plt.bar(counts.keys(), counts.values(), color=['#4C72B0', '#DD8452']) 

plt.xlabel("Classes")
plt.ylabel("Nombre d'images")
plt.title("Distribution des images par classe dans le dataset d'entraînement")
plt.show()