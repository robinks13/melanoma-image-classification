from dataset import MelonaDataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

transform_base = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Créer les datasets (train et test joue le role de validation)
train_dataset = MelonaDataset("melanoma-cancer-dataset/train", transform=transform_base)
val_dataset = MelonaDataset("melanoma-cancer-dataset/test", transform=transform_base)

print(f"Taille du train set : {len(train_dataset)}")
print(f"Taille du val set : {len(val_dataset)}")
print(f"Classes : {train_dataset.classes}")
print(f"Mapping classe->entier : {train_dataset.class_to_idx}")

# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Vérifier la forme d'un batch
images_batch, labels_batch = next(iter(train_loader))
print(f"Forme d'un batch d'images : {images_batch.shape}") # Attendu : [32, 3, 128, 128]
print(f"Forme des labels : {labels_batch.shape}") # Attendu : [32]

def afficher_grille_batch(loader, dataset, titre_global):
    """
    Récupère un batch depuis le loader et affiche les 8 premières images dans une grille 2x4.
    """
    # 1. Récupérer le premier batch d'images et de labels
    images_batch, labels_batch = next(iter(loader))
    
    # 2. Créer une grille de 2 lignes et 4 colonnes
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(titre_global, fontsize=16, fontweight='bold')
    
    # Aplatir le tableau d'axes pour itérer facilement dessus (de 0 à 7)
    axes = axes.flatten()
    
    for i in range(8):
        # PyTorch utilise le format [Canaux, Hauteur, Largeur] (ex: [3, 128, 128])
        # Matplotlib a besoin du format [Hauteur, Largeur, Canaux] (ex: [128, 128, 3])
        # On utilise donc .permute(1, 2, 0) pour réorganiser les dimensions, puis .numpy()
        img_a_afficher = images_batch[i].permute(1, 2, 0).numpy()
        
        # Récupérer l'index du label (entier) puis le nom de la classe correspondant
        label_idx = labels_batch[i].item()
        nom_classe = dataset.classes[label_idx]
        
        # Afficher l'image dans le sous-graphique
        axes[i].imshow(img_a_afficher)
        axes[i].set_title(nom_classe)
        axes[i].axis("off") # Masquer les axes pour un rendu plus propre
        
    plt.tight_layout()
    plt.show()

afficher_grille_batch(train_loader, train_dataset, titre_global="Batch d'Entraînement (Train)")
afficher_grille_batch(val_loader, val_dataset, titre_global="Batch de Validation (Val)")