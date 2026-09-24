# Melanoma Image Classification

An educational computer vision project that compares a custom convolutional neural network (CNN) with a fine-tuned ResNet-18 to classify dermoscopic images as **benign** or **malignant**. The project also includes a small Flask interface for experimenting with single-image predictions.

> **Research/education only.** This model is not a medical device and must not be used to diagnose, screen, or guide treatment. Its predictions and confidence scores are not clinically validated.

## Project overview

- Explore and preprocess a two-class image dataset with PyTorch and torchvision.
- Train a custom CNN and experiment with transfer learning using ResNet-18.
- Track training and validation loss/accuracy and inspect a confusion matrix.
- Run an optional Flask demo that accepts an image and displays the model output.

The repository contains the training and analysis code, plots, and presentation material. The image dataset and model checkpoints are excluded from version control; see [Data and model files](#data-and-model-files). Performance numbers are intentionally not reported here because the original dataset split and evaluation protocol have not yet been independently verified.

## Repository contents

| File | Purpose |
| --- | --- |
| `main.py` | Train the custom CNN and save learning curves. |
| `model.py`, `train.py` | CNN architecture and reusable training/evaluation loops. |
| `dataset.py`, `transforms.py` | Image loading, preprocessing, normalization, and augmentation. |
| `partie*.py`, `projet_cnn.py` | Course exercises and experiments for data exploration and model development. |
| `detect.py` | Load the ResNet-18 checkpoint and predict a single image. |
| `appli/` | Small Flask website, HTML/CSS templates, and PDF view template. |
| `appli/mes_slides.pdf` | Project presentation/report slides (also served by the Flask site). |
| `courbes_cnn_simple.png`, `courbes_resnet.png`, `matrice_confusion.png` | Existing experiment visualizations. |

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install a PyTorch build appropriate for your operating system and hardware from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data and model files

Place the dataset in this structure (the folders become the class labels):

```text
melanoma-cancer-dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

The dataset is not included. Obtain it only from a source whose usage terms permit your intended use, and document the exact source, license, and split before publishing or reporting results. Do not commit patient-identifiable or otherwise restricted data.

The trained ResNet-18 checkpoint (`resnet18_melanoma_final.pth`) is also not included because it is a large binary file. To reproduce predictions, provide a compatible checkpoint at the path expected by the script. The Flask demo expects the checkpoint in `appli/resnet18_melanoma_final.pth`.

## Run the experiments

From this project directory, after placing the dataset as shown above:

```bash
python main.py
```

`main.py` trains the custom CNN for 20 epochs and displays/saves its learning curves. The analysis scripts are course experiments and some are exploratory rather than a single polished, end-to-end training pipeline. Review each script's data paths and transforms before running it. In particular, confirm that validation data is drawn from the intended `test` split and that no test data is used for model selection.

## Run the web demo

After installing dependencies and placing the checkpoint under `appli/`:

```bash
cd appli
python app.py
```

Open the local address printed by Flask and upload an image. The demo runs inference on the CPU. Do not expose it publicly or submit real patient images.

## Limitations

- This is a course project, not a clinically validated model.
- The dataset source, license, size, class balance, and split methodology still need to be documented by the project author.
- The demo's confidence score is a softmax output, not a calibrated probability of disease.
- Results may not generalize across devices, populations, image acquisition settings, or clinical settings.
- Code paths and image-size assumptions vary among exploratory scripts; check configuration before training or inference.

## Français

Projet pédagogique de vision par ordinateur comparant un CNN développé pour le projet et un ResNet-18 ajusté pour classer des images de lésions cutanées en deux catégories : **bénigne** ou **maligne**. Une interface Flask permet également de tester une image.

**Ce projet n'est pas un dispositif médical.** Il ne doit pas servir au dépistage, au diagnostic ou à une décision de traitement. Le jeu de données et les poids entraînés ne sont pas inclus ; leur provenance, leur licence et le protocole d'évaluation doivent être précisés avant toute publication de résultats.

Voir les sections ci-dessus pour l'installation, la structure attendue des données et le lancement des expériences (`python main.py`) ou de la démo (`cd appli && python app.py`).
