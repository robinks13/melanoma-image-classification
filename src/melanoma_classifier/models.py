"""Model architectures used in the experiments."""

from torch import nn
from torchvision import models as vision_models


class SimpleCNN(nn.Module):
    """Compact baseline CNN for 128-pixel RGB images."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=0.5), nn.Linear(64, num_classes)
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(name: str, num_classes: int):
    """Create a project model without downloading external pretrained weights."""
    if name == "simple-cnn":
        return SimpleCNN(num_classes)
    if name == "resnet18":
        model = vision_models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model {name!r}; choose 'simple-cnn' or 'resnet18'.")
