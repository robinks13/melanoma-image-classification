# Melanoma Image Classification

An educational computer-vision project that compares a compact CNN with ResNet-18 for binary classification of dermoscopic images. The repository is organized as a small Python package with reusable data, model, training, and inference modules, plus command-line workflows and a Flask demo.

> **Educational use only.** This model is not a medical device and has not been clinically validated. Its class scores must not be used for screening, diagnosis, or treatment decisions.

## Architecture

```text
src/melanoma_classifier/   Reusable package: data, transforms, models, training, inference
scripts/                   Training and single-image prediction entry points
web/                       Flask demo, templates, and styles
reports/figures/           Existing learning curves and confusion matrix
docs/                      Course presentation
```

The image dataset and model checkpoints are excluded from version control. Existing experiment figures are included as project artifacts; metrics are not presented here as validated results because the dataset source and split protocol have not yet been verified.

## Setup

Use Python 3.10 or newer. Create a virtual environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU support, first install the PyTorch build that matches your operating system and CUDA version using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Dataset layout

Place the dataset outside version control using class-named folders:

```text
data/melanoma/
├── train/
│   ├── Benign/
│   └── Malignant/
└── val/
    ├── Benign/
    └── Malignant/
```

Each split must use the same class folder names. The loader assigns labels in alphabetical order. Before reporting performance, document the dataset source and license, keep patient-level groups separated between train and validation, and evaluate once on a held-out test set.

## Train

```bash
python scripts/train.py --data-dir data/melanoma --model resnet18 --epochs 20
```

Use `--model simple-cnn` for the baseline. The script reports per-epoch loss and accuracy and saves the best validation checkpoint to `artifacts/best_model.pth` by default. Training uses the project’s 128 × 128 preprocessing and training-only augmentation.

## Predict one image

```bash
python scripts/predict.py path/to/image.jpg --checkpoint artifacts/best_model.pth
```

The prediction score is the model’s softmax output, not a calibrated probability.

## Run the Flask demo

After placing a compatible checkpoint at `artifacts/best_model.pth` (or setting `MELANOMA_CHECKPOINT` to its path):

```bash
python web/app.py
```

Open the local address printed by Flask. The demo runs on the CPU by default and limits uploads to 10 MB. Do not expose it publicly or upload patient images.

## Existing project artifacts

- [CNN learning curves](reports/figures/cnn-learning-curves.png)
- [ResNet-18 learning curves](reports/figures/resnet18-learning-curves.png)
- [Confusion matrix](reports/figures/confusion-matrix.png)
- [Course presentation](docs/project-presentation.pdf)

## Limitations

- The dataset provenance, license, sample count, and split methodology still need to be confirmed by the project author.
- Results may not generalize across populations, imaging equipment, or clinical settings.
- A confidence score is not a clinical risk estimate.
- This repository is a course project and is not suitable for medical use.

## Français

Projet pédagogique de vision par ordinateur comparant un CNN compact à ResNet-18 pour classer des images dermoscopiques en deux catégories. Le dépôt sépare le code réutilisable, les scripts d’entraînement et de prédiction, la démo Flask, les figures et le rapport.

Le jeu de données et les poids du modèle ne sont pas inclus. Le modèle n’est pas validé cliniquement et ne doit jamais être utilisé pour le dépistage, le diagnostic ou le choix d’un traitement. Consulte les instructions ci-dessus pour l’installation et l’organisation des données.
