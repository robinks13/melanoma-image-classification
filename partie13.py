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
# (Assurez-vous d'avoir aussi importé vos dataloaders et vos fonctions train/evaluate)

train_dataset_aug = MelonaDataset("melanoma-cancer-dataset/train", transform=train_transform_aug)
val_dataset_aug = MelonaDataset("melanoma-cancer-dataset/train", transform=val_transform)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=512, shuffle=True, num_workers=10)
val_loader_aug = DataLoader(val_dataset_aug, batch_size=512, shuffle=False, num_workers=10)

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

def evaluer_modele_final(model, val_loader, device, class_names):
    """
    Parcourt le val_loader, génère la matrice de confusion et le rapport de classification.
    """
    print("\n🔍 Évaluation finale en cours...")
    model.eval() # Mode évaluation
    
    y_true = []
    y_pred = []
    
    # 1. Parcourir le val_loader sans calculer les gradients
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = outputs.argmax(dim=1) # On prend l'indice de la probabilité max
            
            # 2. Convertir en listes CPU
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            
    # 3. Calculs via scikit-learn
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    print("\n📊 --- RAPPORT DE CLASSIFICATION ---")
    print(report)
    
    # 4. Affichage de la matrice avec plt.imshow et annotations
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Matrice de Confusion - ResNet18", fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # Configuration des axes
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=12)
    plt.yticks(tick_marks, class_names, fontsize=12)
    
    # Annotation des cellules
    seuil = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > seuil else "black",
                     fontsize=12, fontweight='bold')
            
    plt.ylabel('Vrais Labels (Réalité)', fontsize=12)
    plt.xlabel('Prédictions du Modèle', fontsize=12)
    plt.tight_layout()
    plt.savefig("matrice_confusion.png", dpi=150)
    plt.show()

def visualiser_cas_etude(model, dataset, device, class_names):
    """
    Collecte et affiche 8 erreurs (4 de chaque classe) 
    et 8 succès (4 de chaque classe).
    """
    model.eval() # 🚨 INDISPENSABLE : retire le dropout
    
    # On crée 4 listes distinctes pour bien équilibrer
    erreurs_benign = []
    erreurs_malignant = []
    succes_benign = []
    succes_malignant = []
    
    # Valeurs de normalisation de la Partie 6
    MEAN = torch.tensor([0.7229, 0.5555, 0.5390]).view(3, 1, 1)
    STD = torch.tensor([0.1875, 0.1964, 0.2101]).view(3, 1, 1)

    print("🔎 Recherche d'exemples équilibrés dans le dataset de validation...")

    with torch.no_grad():
        # On parcourt le dataset image par image
        for i in range(len(dataset)):
            img, label = dataset[i]
            
            # Prédiction
            output = model(img.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            
            # Noms des classes ('benign' ou 'malignant')
            vrai_nom = class_names[label].lower()
            pred_nom = class_names[pred].lower()
            
            # Dénormalisation pour l'affichage (remet en [0, 1])
            img_visu = img * STD + MEAN
            img_visu = img_visu.clamp(0, 1).permute(1, 2, 0).numpy()
            
            info = {
                'img': img_visu,
                'vrai': class_names[label],
                'pred': class_names[pred]
            }

            # Tri dans les 4 catégories
            if pred != label: # Cas d'erreur
                if vrai_nom == 'benign' and len(erreurs_benign) < 4:
                    erreurs_benign.append(info)
                elif vrai_nom == 'malignant' and len(erreurs_malignant) < 4:
                    erreurs_malignant.append(info)
            else: # Cas de succès
                if vrai_nom == 'benign' and len(succes_benign) < 4:
                    succes_benign.append(info)
                elif vrai_nom == 'malignant' and len(succes_malignant) < 4:
                    succes_malignant.append(info)
                
            # On s'arrête dès que TOUTES nos listes ont atteint 4 images
            if len(erreurs_benign) == 4 and len(erreurs_malignant) == 4 and \
               len(succes_benign) == 4 and len(succes_malignant) == 4:
                break

    # On fusionne les listes pour l'affichage (4 de chaque)
    erreurs_totales = erreurs_benign + erreurs_malignant
    succes_totaux = succes_benign + succes_malignant

    # Affichage des Erreurs
    tracer_grille(erreurs_totales, "⚠️ Erreurs : 4 Faux Positifs (haut) et 4 Faux Négatifs (bas)")
    
    # Affichage des Succès
    tracer_grille(succes_totaux, "✅ Succès : 4 Vrais Bénins (haut) et 4 Vrais Malins (bas)")

def tracer_grille(liste_info, titre_general):
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    
    # y=0.98 remonte légèrement le titre principal pour libérer de la place
    fig.suptitle(titre_general, fontsize=16, fontweight='bold', 
                 color='red' if "Erreurs" in titre_general else 'green', y=0.98)
    
    for i, info in enumerate(liste_info):
        ax = axes[i//4, i%4]
        ax.imshow(info['img'])
        color = 'red' if info['vrai'] != info['pred'] else 'green'
        
        # J'ai passé la taille de la police à 11 pour que ce soit un peu plus lisible
        ax.set_title(f"Vrai: {info['vrai']}\nPred: {info['pred']}", color=color, fontsize=11)
        ax.axis('off')
        
    plt.tight_layout()
    # LA LIGNE MAGIQUE : hspace=0.4 force un grand espace blanc vertical entre les lignes
    plt.subplots_adjust(hspace=0.4, top=0.88)
    
    plt.show() 

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
        transforms.Resize((128, 128)), # Mettez la taille que vous avez utilisée dans le TP
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

if __name__ == '__main__':
    # 1. Vérification du GPU local
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device utilisé en local : {device}")
    
    # Si ça affiche "cpu", c'est que votre PyTorch local n'a pas été installé avec CUDA !
    
    # 2. Charger ResNet18 pré-entraîné
    print("Téléchargement de ResNet18...")
    resnet_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in resnet_ft.parameters():
        param.requires_grad = False

    num_classes = len(train_dataset_aug.classes)
    resnet_ft.fc = nn.Linear(resnet_ft.fc.in_features, num_classes)
    resnet_ft = resnet_ft.to(device)
    
    # Degeler layer4 + fc

    for param in resnet_ft.layer4.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in resnet_ft.parameters())
    trainable_params = sum(p.numel() for p in resnet_ft.parameters() if p.requires_grad)

    print("-" * 50)
    print(f"🧠 Total des paramètres : {total_params:,}")
    print(f"🔥 Paramètres entraînables : {trainable_params:,}")
    print(f"❄️ Paramètres gelés : {total_params - trainable_params:,}")
    print("-" * 50)
    
    # 5. Configuration (On optimise UNIQUEMENT la nouvelle couche fc)
    criterion = nn.CrossEntropyLoss()
    # LR plus faible pour les couches pre-entrainee, plus eleve pour la tete
    optimizer_ft = optim.Adam([
        {"params": resnet_ft.layer4.parameters(), "lr": 1e-4},
        {"params": resnet_ft.fc.parameters(), "lr": 1e-3},
    ])

    NUM_EPOCHS = 20
    history_resnet = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("🚀 Démarrage de l'entraînement de ResNet18 en local...")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        
        # On utilise le loader avec augmentation
        train_loss, train_acc = train_one_epoch(resnet_ft, train_loader_aug, criterion, optimizer_ft, device)
        val_loss, val_acc = evaluate(resnet_ft, val_loader_aug, criterion, device)
        
        duree = time.time() - t0
        
        history_resnet["train_loss"].append(train_loss)
        history_resnet["val_loss"].append(val_loss)
        history_resnet["train_acc"].append(train_acc)
        history_resnet["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss val : {val_loss:.4f} | "
              f"Acc train : {train_acc:.3f} | Acc val : {val_acc:.3f} | "
              f"(duree: {duree:.1f} s)")

    # 6. Tracez et sauvegardez les courbes (assurez-vous d'avoir la fonction)
    tracer_courbes(history_resnet, titre="Transfer Learning (ResNet18)", save_name="courbes_resnet.png")
    evaluer_modele_final(resnet_ft, val_loader_aug, device, train_dataset_aug.classes)
    visualiser_cas_etude(resnet_ft, val_dataset_aug, device, train_dataset_aug.classes)
    predire_nouvelle_image("lenny.jpeg", resnet_ft, train_dataset_aug.classes, device)
    PATH = "resnet18_melanoma_final.pth"

    # On enregistre uniquement les poids (le state_dict)
    torch.save(resnet_ft.state_dict(), PATH)

    print(f"✅ Modèle sauvegardé avec succès sous : {PATH}")