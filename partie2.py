import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch

TRAIN_DIR = "melanoma-cancer-dataset/train"

# =====================================================================
# 3.1. Inspecter une image
# =====================================================================

# Charger une image (adaptez le chemin)
chemin = os.path.join(TRAIN_DIR, "Benign",os.listdir(os.path.join(TRAIN_DIR,"Benign")) [0])
img_pil = Image.open(chemin).convert("RGB")
print(f"Tailleoriginale (PIL) : {img_pil.size}") # format PIL :(largeur, hauteur)
print(f"Type des pixels PIL : {type(img_pil.getpixel((0,0)))}") # tuple d’entiers 0-255

# Convertiren tenseur PyTorch
to_tensor =transforms.ToTensor()
img_tensor =to_tensor(img_pil)

print(f"Formedu tenseur : {img_tensor.shape}") # [C, H, W]
print(f"Valeurmin : {img_tensor.min():.4f}")
print(f"Valeurmax : {img_tensor.max():.4f}")

classes = sorted(os.listdir(TRAIN_DIR))

# =====================================================================
# 3.2 Visualiser les canaux RGB séparément
# =====================================================================

print("\nPréparation de la visualisation des canaux RGB...")

# 1. Récupérer une image d'exemple (la toute première du dossier de la première classe)
chemin_dossier_exemple = os.path.join(TRAIN_DIR, classes[0])
nom_fichier_exemple = os.listdir(chemin_dossier_exemple)[0]
chemin_image_exemple = os.path.join(chemin_dossier_exemple, nom_fichier_exemple)

image_pil = Image.open(chemin_image_exemple)

# 2. Convertir l'image PIL en tenseur PyTorch
# ToTensor() convertit automatiquement l'image en format [Canaux, Hauteur, Largeur]
# et normalise les pixels entre 0 et 1.
transform = transforms.ToTensor()
img_tensor = transform(image_pil)

# 3. Préparer la figure Matplotlib (1 ligne, 4 colonnes)
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes = axes.flatten()

# --- Affichage 1 : L'image originale ---
# On réordonne les dimensions [C, H, W] -> [H, W, C] avec permute pour matplotlib
img_originale = img_tensor.permute(1, 2, 0).numpy()
axes[0].imshow(img_originale)
axes[0].set_title("Image Originale")
axes[0].axis("off")

# --- Affichage 2 : Canal Rouge (Index 0) ---
# img_tensor[0] extrait la matrice 2D [H, W] du rouge
canal_rouge = img_tensor[0].numpy()
axes[1].imshow(canal_rouge, cmap='Reds')
axes[1].set_title("Canal Rouge (R)")
axes[1].axis("off")

# --- Affichage 3 : Canal Vert (Index 1) ---
# img_tensor[1] extrait la matrice 2D [H, W] du vert
canal_vert = img_tensor[1].numpy()
axes[2].imshow(canal_vert, cmap='Greens')
axes[2].set_title("Canal Vert (G)")
axes[2].axis("off")

# --- Affichage 4 : Canal Bleu (Index 2) ---
# img_tensor[2] extrait la matrice 2D [H, W] du bleu
canal_bleu = img_tensor[2].numpy()
axes[3].imshow(canal_bleu, cmap='Blues')
axes[3].set_title("Canal Bleu (B)")
axes[3].axis("off")

# On récupère la shape exacte du tenseur (ex: [3, 224, 224])
forme_tenseur = list(img_tensor.shape) 
plt.suptitle(f"Forme du tenseur : {forme_tenseur}", fontsize=16, fontweight='bold')
# Ajustement pour un affichage propre
plt.tight_layout()
plt.show()