import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# ============================================================
# MODEL DEFINITION (matches your training architecture)
# ============================================================

class CNN_A(nn.Module):
    """Simple CNN with L2 regularization"""
    def __init__(self, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * (img_size // 4) * (img_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Binary classification
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# LOAD MODEL + WEIGHTS
# ============================================================

def load_model(weights_path, device='cpu', img_size=128):
    model = CNN_A(img_size=img_size)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ============================================================
# PREPROCESSING
# ============================================================

def get_transform(img_size=128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# ============================================================
# INFERENCE FUNCTION
# ============================================================

def predict_image(image_path, model, device, img_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        pred = 1 if prob >= 0.6 else 0  # Threshold for binary classification
        label = "CORROSION" if pred == 0 else "NOCORROSION"

    return label, prob
# ===========================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Set these according to your environment
    
    weights_path = "/data/ashish/SAITEJAMS/AI_IN_MAN_Project/trained_models_custom_cnn/CNN_A_normal_best.pth"
    test_image_path = "/data/ashish/SAITEJAMS/AI_IN_MAN_Project/Dataset/Denoised/CORROSION/000001.jpg"  # change to your test image
    img_size = 128  # must match training size

    device = 'cpu'
    model = load_model(weights_path, device, img_size)

    label, prob = predict_image(test_image_path, model, device, img_size)
    
    if label == "CORROSION":
        prob = 1-prob
    print(f"Prediction: {label}, Confidence: {prob}")

