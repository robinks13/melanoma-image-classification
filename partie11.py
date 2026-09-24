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
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 3. GELER les couches existantes (On ne touche pas au "cerveau" visuel)
    for param in resnet.parameters():
        param.requires_grad = False

    # 4. Remplacer la dernière couche pour nos mélanomes
    num_ftrs = resnet.fc.in_features
    # Remplacez "len(train_dataset_aug.classes)" par 2 si vous connaissez déjà le nombre
    num_classes = len(train_dataset_aug.classes) 
    resnet.fc = nn.Linear(num_ftrs, num_classes)

    # On envoie le modèle sur votre GPU
    resnet = resnet.to(device)

    total_params = sum(p.numel() for p in resnet.parameters())
    trainable_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)

    print("-" * 50)
    print(f"🧠 Total des paramètres : {total_params:,}")
    print(f"🔥 Paramètres entraînables : {trainable_params:,}")
    print(f"❄️ Paramètres gelés : {total_params - trainable_params:,}")
    print("-" * 50)
    
    # 5. Configuration (On optimise UNIQUEMENT la nouvelle couche fc)
    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.Adam(resnet.fc.parameters(), lr=1e-5)

    NUM_EPOCHS = 20
    history_resnet = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("🚀 Démarrage de l'entraînement de ResNet18 en local...")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        
        # On utilise le loader avec augmentation
        train_loss, train_acc = train_one_epoch(resnet, train_loader_aug, criterion, optimizer_resnet, device)
        val_loss, val_acc = evaluate(resnet, val_loader_aug, criterion, device)
        
        duree = time.time() - t0
        
        history_resnet["train_loss"].append(train_loss)
        history_resnet["val_loss"].append(val_loss)
        history_resnet["train_acc"].append(train_acc)
        history_resnet["val_acc"].append(val_acc)
        
        # Affichage VRAM compatible local
        vram_info = ""
        if device.type == 'cuda':
            pic_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            vram_info = f" | PIC VRAM: {pic_vram:.2f} GB"

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Loss val : {val_loss:.4f} | "
              f"Acc train : {train_acc:.3f} | Acc val : {val_acc:.3f} | "
              f"(duree: {duree:.1f} s){vram_info}")

    # 6. Tracez et sauvegardez les courbes (assurez-vous d'avoir la fonction)
    tracer_courbes(history_resnet, titre="Transfer Learning (ResNet18)", save_name="courbes_resnet.png")