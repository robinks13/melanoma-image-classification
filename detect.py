from dataset import MelonaDataset
from transforms import train_transform_aug, val_transform
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from train import train_one_epoch, evaluate

import torch.nn.functional as F
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset_aug = MelonaDataset("melanoma-cancer-dataset/train", transform=train_transform_aug)
val_dataset_aug = MelonaDataset("melanoma-cancer-dataset/train", transform=val_transform)

def charger_mon_modele(chemin_weights, num_classes, device):
    # 1. On recrée l'architecture exacte (ResNet18)
    model = models.resnet18(weights=None) # Pas besoin de télécharger les poids ImageNet ici
    
    # 2. On remet la même couche finale que pendant l'entraînement
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # 3. On charge les poids sauvegardés
    # Le map_location permet de charger sur CPU même si on a entraîné sur GPU
    model.load_state_dict(torch.load(chemin_weights, map_location=device))
    
    model.to(device)
    model.eval() # Mode évaluation par défaut
    print(f"🚀 Modèle chargé et prêt à l'emploi !")
    return model

# Exemple d'utilisation :
resnet_ft = charger_mon_modele("resnet18_melanoma_final.pth", 2, device)

def predire_nouvelle_image(chemin_image, model, class_names, device):
    """
    Ouvre une image, la prépare, la fait passer dans le modèle 
    et renvoie la prédiction avec un score de confiance.
    """
    print(f"\n🔍 Analyse de l'image : {chemin_image}")
    
    # 1. On fige le modèle (très important !)
    model.eval()
    
    # 2. Préparation de l'image (Identique à val_loader_norm !)
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)), # Mettez la taille que vous avez utilisée dans le TP
        transforms.ToTensor(),
        # Utilisez VOS valeurs de normalisation de la Partie 6
        transforms.Normalize(mean=[0.7229, 0.5555, 0.5390], 
                             std=[0.1875, 0.1964, 0.2101]) 
    ])
    
    # 3. Ouverture et transformation
    try:
        image_pil = Image.open(chemin_image).convert('RGB')
    except Exception as e:
        print(f"❌ Erreur lors de l'ouverture de l'image : {e}")
        return
        
    image_tensor = transform_test(image_pil)
    
    # Le modèle attend un "batch" d'images : [Batch, Canaux, Hauteur, Largeur]
    # On ajoute donc une fausse dimension de batch avec unsqueeze(0) -> [1, 3, 128, 128]
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    # 4. Inférence (sans calculer les gradients)
    with torch.no_grad():
        sortie_brute = model(image_batch)
        
        # On passe les résultats bruts dans un Softmax pour avoir des pourcentages
        probabilites = F.softmax(sortie_brute, dim=1)[0] * 100
        
        # On trouve la classe qui a le plus haut pourcentage
        index_pred = probabilites.argmax().item()
        classe_predite = class_names[index_pred]
        confiance = probabilites[index_pred].item()

    # 5. Affichage du résultat
    plt.figure(figsize=(5, 5))
    plt.imshow(image_pil)
    plt.axis('off')
    
    couleur = 'red' if classe_predite.lower() == 'malignant' else 'green'
    titre = f"Prédiction : {classe_predite}\nConfiance : {confiance:.1f}%"
    plt.title(titre, color=couleur, fontsize=14, fontweight='bold')
    
    # Affichage des probabilités détaillées dans la console
    print("-" * 30)
    for i, nom_classe in enumerate(class_names):
        print(f"Probabilité {nom_classe} : {probabilites[i]:.1f}%")
    print("-" * 30)
    
    plt.show()

predire_nouvelle_image("marco.jpg", resnet_ft, train_dataset_aug.classes, device)