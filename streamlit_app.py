# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle

# -----------------------------
# Define the hybrid model class
# -----------------------------
class HybridNet(nn.Module):
    def __init__(self, num_tabular_features=3, num_classes=7):
        super(HybridNet, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, tabular):
        img_feat = self.cnn(image)
        tab_feat = self.tabular_net(tabular)
        combined = torch.cat((img_feat, tab_feat), dim=1)
        out = self.classifier(combined)
        return out

# -----------------------------
# Load model + encoders
# -----------------------------
@st.cache_resource
def load_model_and_encoders():
    model = HybridNet()
    model.load_state_dict(torch.load("hybrid_model.pth", map_location="cpu"))
    model.eval()
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_model_and_encoders()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Skin Lesion Classifier (HAM10000 Hybrid AI)")
st.write("Upload a lesion image and enter details to predict the diagnosis.")

# Input: Image
uploaded_image = st.file_uploader("📷 Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# Input: Metadata
age = st.slider("📅 Age", 0, 100, 30)
sex = st.selectbox("⚧️ Sex", ["male", "female", "unknown"])
site = st.selectbox("📍 Body site", encoders['anatom_site_general_challenge'].classes_)

# Predict Button
if uploaded_image and st.button("🔍 Predict"):
    # Transform image
    img = Image.open(uploaded_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image_tensor = transform(img).unsqueeze(0)

    # Encode metadata
    sex_encoded = encoders['sex'].transform([sex])[0]
    site_encoded = encoders['anatom_site_general_challenge'].transform([site])[0]
    tabular_tensor = torch.tensor([[age, sex_encoded, site_encoded]], dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output = model(image_tensor, tabular_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = encoders['diagnosis'].inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()

    st.image(img, caption="Uploaded Lesion", use_container_width=True)
    st.success(f"🧬 **Predicted Diagnosis**: `{pred_class.upper()}`")
    st.info(f"🧪 Confidence: **{confidence*100:.2f}%**")
