"""Train a CNN or ResNet-18 on a class-folder dataset."""

import argparse
from pathlib import Path

import torch
from torch import nn, optim

from melanoma_classifier.data import make_dataloaders
from melanoma_classifier.models import build_model
from melanoma_classifier.training import run_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/melanoma"))
    parser.add_argument("--model", choices=("simple-cnn", "resnet18"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/best_model.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_dataloaders(
        args.data_dir, args.batch_size, args.workers
    )
    model = build_model(args.model, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0

    print(f"Device: {device}; classes: {classes}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.3f}"
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                },
                args.output,
            )
    print(f"Best validation accuracy: {best_accuracy:.3f}; checkpoint: {args.output}")


if __name__ == "__main__":
    main()
