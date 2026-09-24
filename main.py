from dataset import MelonaDataset
from transforms import transform_base, transform_normalise, MEAN, STD
from model import SimpleCNN, compter_parametres
from train import train_one_epoch, evaluate

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Créer les datasets (train et test joue le role de validation)
train_dataset = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_base)
val_dataset = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_base)

train_dataset_norm = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_normalise)
val_dataset_norm = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_normalise)

train_loader_norm = DataLoader(train_dataset_norm, batch_size=32, shuffle=True, num_workers=0)
val_loader_norm = DataLoader(val_dataset_norm, batch_size=32, shuffle=False, num_workers=0)

# Récupérer la toute première image (index 0) dans les deux versions
img_base, label_base = train_dataset[0]
img_norm, label_norm = train_dataset_norm[0]

# Formule pour ramener les pixels entre 0 et 1 pour l'affichage
def denormalize_for_display(img_tensor):
    img_min = img_tensor.min()
    img_max = img_tensor.max()
    return (img_tensor - img_min) / (img_max - img_min)

# Affichage côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Image classique
ax1.imshow(img_base.permute(1, 2, 0).numpy())
ax1.set_title(f"Avant Normalisation\nMin: {img_base.min():.2f}, Max: {img_base.max():.2f}")
ax1.axis("off")

# Image normalisée (et corrigée pour l'affichage)
img_norm_visu = denormalize_for_display(img_norm)
ax2.imshow(img_norm_visu.permute(1, 2, 0).numpy())
ax2.set_title(f"Après Normalisation\nMin: {img_norm.min():.2f}, Max: {img_norm.max():.2f}")
ax2.axis("off")

plt.suptitle("Effet de la normalisation sur une image", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

num_classes = len(train_dataset_norm.classes)

# Instancier le modèle et l'envoyer sur le device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device utilisé : {device}")
model = SimpleCNN(num_classes=num_classes).to(device)
compter_parametres(model)

# Configuration de l'entraînement
NUM_EPOCHS = 20

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3) # Adam recommandé pour commencer

# Historique pour les courbes
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

print("Démarrage de l'entraînement...")

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader_norm, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader_norm, criterion, device)
    duree = time.time() - t0
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
          f"Loss train : {train_loss:.4f} | Loss val : {val_loss:.4f} | "
          f"Acc train : {train_acc:.3f} | Acc val : {val_acc:.3f} | "
          f"(duree: {duree:.1f} s)")

def tracer_courbes(history, titre="CNN simple", save_name="courbes_cnn_simple.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    
    ax1.plot(epochs, history["train_loss"], label="Train", color='steelblue')
    ax1.plot(epochs, history["val_loss"], label="Validation", color='tomato')
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - {titre}"); ax1.legend(); ax1.grid(alpha=0.3)
    
    # Courbe Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", color='steelblue')
    ax2.plot(epochs, history["val_acc"], label="Validation", color='tomato')
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy - {titre}"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()

tracer_courbes(history, titre="CNN simple")