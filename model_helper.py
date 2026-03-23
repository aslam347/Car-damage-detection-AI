import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st   # ✅ ADDED

trained_model = None
device = torch.device("cpu")

class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]

# ------------------ MODEL ------------------
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        # Freeze layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# ✅ ADDED: cache model loading (NO CHANGE to your logic)
@st.cache_resource
def get_model():
    model = CarClassifierResNet()
    model.load_state_dict(
        torch.load("model\\saved_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model


# ------------------ PREDICT FUNCTION ------------------
def predict(image_path):
    global trained_model

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # ✅ ONLY CHANGE: use cached model
    if trained_model is None:
        trained_model = get_model()

    with torch.no_grad():
        output = trained_model(image_tensor)

        probs = torch.softmax(output, dim=1)
        confidence = torch.max(probs).item()

        _, predicted_class = torch.max(output, 1)

        return class_names[predicted_class.item()], confidence, probs[0].tolist()