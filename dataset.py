import os
from PIL import Image
from torch.utils.data import Dataset

class MelonaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = [] # liste de tuples (chemin_image, label_entier)

        # Les classes sont les noms des sous-dossiers
        self.classes = sorted(os.listdir(data_dir))
        # Associer chaque nom de classe a un entier : {"Benign": 0, "Malignant": 1}
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Parcourir chaque classe et lister les images
        for classe in self.classes:
            label = self.class_to_idx[classe]
            dossier = os.path.join(data_dir, classe)
            for nom_fichier in os.listdir(dossier):
                if nom_fichier.endswith(".jpg"):
                    chemin = os.path.join(dossier, nom_fichier)
                    self.samples.append((chemin, label))
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chemin, label = self.samples[idx]
        image = Image.open(chemin).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


'''
if __name__ =="__main__":
    ds = MelonaDataset("melanoma-cancer-dataset/train")
    print(len(ds), ds.classes) 
    img, label =ds[0] 
    print(img.size,label)
'''