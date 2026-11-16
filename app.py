import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

# --- Architecture Definition (Required for standalone app) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(attention))

class EfficientNetWithSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.feature_extractor = self.backbone.extract_features
        self.spatial_attention = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1792, 1)
        self.sigmoid = nn.Sigmoid()
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.spatial_attention(x)
        
        # Register hook for Grad-CAM Explainability
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.avgpool(x).flatten(1)
        return self.sigmoid(self.fc(x))

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.title("Deepfake Detection with Spatial Attention Explainability")
st.markdown("Developed by **Sai Nithin Battu** | MSc Artificial Intelligence, Teesside University")

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientNetWithSAM().to(device)
    # Ensure your trained weights file is named correctly
    try:
        model.load_state_dict(torch.load('efficientnet_sam_c23.pth', map_location=device))
    except FileNotFoundError:
        st.warning("Model weights 'efficientnet_sam_c23.pth' not found. Please upload them to run inference.")
    model.eval()
    mtcnn = MTCNN(image_size=380, margin=40, keep_all=False, device=device)
    return model, mtcnn, device

model, mtcnn, device = load_models()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2, col3 = st.columns(3)
    img = Image.open(uploaded_file).convert('RGB')
    col1.image(img, caption="Original Input", use_column_width=True)

    with st.spinner("Extracting face and analyzing spatial attention..."):
        face_tensor = mtcnn(img)
