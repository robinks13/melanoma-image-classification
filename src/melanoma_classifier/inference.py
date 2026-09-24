"""Checkpoint loading and single-image prediction helpers."""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .models import build_model
from .transforms import build_transform


def load_model(checkpoint_path: str | Path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        name = checkpoint.get("model_name", "resnet18")
        classes = checkpoint.get("classes", ["Benign", "Malignant"])
        state = checkpoint["model_state_dict"]
    else:
        name, classes, state = "resnet18", ["Benign", "Malignant"], checkpoint
    model = build_model(name, len(classes))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, device


def predict_image(model, image: Image.Image, classes, device):
    tensor = build_transform()(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = F.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    return classes[index], round(float(probabilities[index].item()) * 100, 1)
