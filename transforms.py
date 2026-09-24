import torch
from torch.utils.data import DataLoader
from dataset import MelonaDataset
import torchvision.transforms as transforms

transform_base = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Créer les datasets (train et test joue le role de validation)
dataset_stats = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_base)

def calculer_mean_std(dataset):
    """Calcule la moyenne et l'écart-type par canal sur le dataset."""
    print("Calcul de MEAN et STD en cours (cela peut prendre quelques secondes)...")
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_batches = 0
    
    for images, _ in loader:
        print("image")
        # On moyenne sur les dimensions batch, H, W (dim 0, 2, 3)
        mean += images.mean(dim=[0, 2, 3])
        std += images.std(dim=[0, 2, 3])
        n_batches += 1
        
    mean /= n_batches
    std /= n_batches
    print(f"MEAN calculée : {mean}")
    print(f"STD calculée : {std}")
    return mean.tolist(), std.tolist()

# Utiliser le dataset de base (déjà en 128x128 via transform_base) pour calculer les stats
#MEAN, STD = calculer_mean_std(dataset_stats)

MEAN = [0.7229, 0.5555, 0.5390]
STD = [0.1875, 0.1964, 0.2101]

transform_normalise = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


# 1. Pipeline d'augmentation pour l'entraînement
train_transform_aug = transforms.Compose([
    #transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transform = transforms.Compose([
    #transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])