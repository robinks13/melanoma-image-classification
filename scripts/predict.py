"""Predict the class of one image using a saved project checkpoint."""

import argparse
from pathlib import Path

from PIL import Image

from melanoma_classifier.inference import load_model, predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/best_model.pth"))
    args = parser.parse_args()
    model, classes, device = load_model(args.checkpoint)
    with Image.open(args.image) as image:
        label, confidence = predict_image(model, image, classes, device)
    print(f"Prediction: {label} ({confidence:.1f}% model confidence)")


if __name__ == "__main__":
    main()
