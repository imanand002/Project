from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Model Architecture
# -----------------------------
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

NUM_CLASSES = 7

device = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))

# Create model structure
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

# Load trained weights
checkpoint = torch.load("best_efficientnet_b0.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

# -----------------------------
# Transform for prediction
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict-image")  # ← CHANGED FROM "/predict" to "/predict-image"
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io. BytesIO(image_bytes)).convert("RGB")

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()

    return {
        "emotion":  EMOTIONS[pred],
        "index": pred
    }
