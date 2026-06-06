import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("🔍 TrustLens AI")
st.write("Upload an image to detect whether it is AI-generated or Real.")

# Load model

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(
torch.load("trustlens_model.pth", map_location="cpu")
)

model.eval()

# Image transform

transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor()
])

# Upload image

uploaded = st.file_uploader(
"Choose an image",
type=["jpg", "jpeg", "png"]
)
# Prediction

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    classes = ["AI Generated", "Real Image"]

    st.success(
        "Prediction: " + classes[pred]
    )