import streamlit as st
import torch
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

st.title("🧬 Skin Cancer Detection App")



model_choice = st.selectbox(
    "Select Model",
    ["EfficientNet-B0", "ResNet-50"]
)

model_path = "models/efficientnet_model.pkl" if model_choice == "EfficientNet-B0" else "models/resnet_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, 1)
        pred_class = probs.argmax(1).item()
        confidence = probs[0][pred_class].item()

    st.write("### Prediction")
    st.write("Model:", model_choice)
    st.write("Class:", class_names[pred_class])
    st.write("Confidence:", round(confidence * 100, 2), "%")



    target_layer = model.features[-1] if model_choice == "EfficientNet-B0" else model.layer4[-1]
    cam = GradCAM(model, target_layer)

    cam_map = cam.generate(input_tensor, pred_class)

    plt.imshow(cam_map, cmap="jet")
    plt.axis("off")

    st.write("### Grad-CAM")
    st.pyplot(plt)
