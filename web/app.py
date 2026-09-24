"""Local Flask demo for the image classifier."""

import os
from pathlib import Path

from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError

from melanoma_classifier.inference import load_model, predict_image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("MELANOMA_CHECKPOINT", ROOT / "artifacts/best_model.pth"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
_loaded = None


def get_model():
    global _loaded
    if _loaded is None:
        _loaded = load_model(CHECKPOINT)
    return _loaded


@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = error = None
    if request.method == "POST":
        uploaded = request.files.get("image")
        if not uploaded or not uploaded.filename:
            error = "Choose an image file first."
        else:
            try:
                model, classes, device = get_model()
                with Image.open(uploaded.stream) as image:
                    result, confidence = predict_image(model, image, classes, device)
            except FileNotFoundError:
                error = f"Model checkpoint not found: {CHECKPOINT}"
            except (UnidentifiedImageError, OSError):
                error = "The uploaded file is not a readable image."
    return render_template("index.html", result=result, confidence=confidence, error=error)


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html", result=None, confidence=None, error="Image must be 10 MB or smaller."
    ), 413


if __name__ == "__main__":
    app.run(debug=False)
