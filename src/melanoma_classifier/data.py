"""Dataset and data-loader helpers."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .transforms import build_transform


def make_dataloaders(data_dir: str | Path, batch_size: int = 32, workers: int = 0):
    """Load ``train`` and ``val`` class-folder splits below ``data_dir``."""
    root = Path(data_dir)
    train_set = ImageFolder(root / "train", transform=build_transform(training=True))
    val_set = ImageFolder(root / "val", transform=build_transform(training=False))
    if train_set.classes != val_set.classes:
        raise ValueError("Train and validation splits must contain the same class folders.")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers
    )
    return train_loader, val_loader, train_set.classes
