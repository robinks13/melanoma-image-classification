from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# ==========================
# MODEL
# ==========================
device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

try:
    model.load_state_dict(torch.load("resnet18_melanoma_final.pth", map_location=device))
except:
    print("Model not found")

model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7229, 0.5555, 0.5390],
                         std=[0.1875, 0.1964, 0.2101])
])

classes = ['Lésion Bénigne', 'Mélanome Malin']


# ==========================
# ROUTES
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            image = Image.open(file).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)[0] * 100

                idx = probs.argmax().item()
                result = classes[idx]
                confidence = round(probs[idx].item(), 1)

    return render_template("index.html", result=result, confidence=confidence)


# ==========================
# RUN LOCAL
# ==========================
if __name__ == "__main__":
    app.run(debug=True)