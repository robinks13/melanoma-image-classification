# Flask Web Demo

This folder contains the Flask interface for trying the trained ResNet-18 classifier on one uploaded image. The checkpoint `resnet18_melanoma_final.pth` is intentionally excluded from the repository because of its size.

## Run locally

1. Install project dependencies from the repository root.
2. Place a compatible checkpoint at `appli/resnet18_melanoma_final.pth`.
3. From this folder, run `python app.py`.

The app runs inference on CPU and is for educational demonstration only. It is not a medical diagnostic tool.
