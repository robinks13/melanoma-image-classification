from dataset import MelonaDataset
from transforms import train_transform_aug, val_transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from train import train_one_epoch, evaluate

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
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
    for param in resnet_ft.layer3.parameters():
        param.requires_grad = True

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