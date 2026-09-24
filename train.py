import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Effectue une epoch d'entraînement. Retourne (loss_moyenne, accuracy)."""
    model.train() # mode entraînement (active Dropout, BatchNorm, etc.)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for images, labels in loader:
        # 1. Envoyer les données sur GPU
        images = images.to(device)
        labels = labels.to(device)
        
        # 2. Remettre les gradients à zéro
        optimizer.zero_grad()
        
        # 3. Passer les images dans le réseau
        outputs = model(images)
        
        # 4. Calculer la loss
        loss = criterion(outputs, labels)
        
        # 5. Backpropagation & mise à jour des poids
        loss.backward()
        optimizer.step()
        
        # Suivi des métriques
        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1) # classe avec le score le plus haut
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
        
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Évalue le modèle sur un loader. Retourne (loss_moyenne, accuracy)."""
    model.eval() # mode évaluation (désactive Dropout, etc.)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad(): # pas de calcul de gradient -> plus rapide
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy