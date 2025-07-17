# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import io

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Dermatology Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for attractive UI
# -----------------------------
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
        animation: headerPulse 4s ease-in-out infinite;
    }

    @keyframes headerPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .floating-objects {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
    }

    .medical-object {
        position: absolute;
        color: rgba(255, 255, 255, 1.5);
        font-size: 1.5rem;
        animation: float 6s ease-in-out infinite;
    }

    .medical-object:nth-child(1) {
        top: 20%;
        left: 10%;
        animation-delay: 0s;
    }

    .medical-object:nth-child(2) {
        top: 60%;
        left: 85%;
        animation-delay: 1s;
    }

    .medical-object:nth-child(3) {
        top: 40%;
        left: 20%;
        animation-delay: 2s;
    }

    .medical-object:nth-child(4) {
        top: 30%;
        left: 70%;
        animation-delay: 3s;
    }

    .medical-object:nth-child(5) {
        top: 70%;
        left: 15%;
        animation-delay: 4s;
    }

    .medical-object:nth-child(6) {
        top: 15%;
        left: 80%;
        animation-delay: 5s;
    }

    @keyframes float {
        0%, 100% { 
            transform: translateY(0px) rotate(0deg);
            opacity: 0.7;
        }
        50% { 
            transform: translateY(-20px) rotate(180deg);
            opacity: 0.9;
        }
    }

    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 10;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    @keyframes titleGlow {
        from { text-shadow: 0 0 10px rgba(255,255,255,0.5); }
        to { text-shadow: 0 0 20px rgba(255,255,255,0.8); }
    }

    .main-header p {
        color: #f0f0f0;
        text-align: center;
        font-size: 1.2rem;
        margin: 0;
        position: relative;
        z-index: 10;
        animation: subtitleFade 2s ease-in-out infinite alternate;
    }

    @keyframes subtitleFade {
        from { opacity: 0.8; }
        to { opacity: 1; }
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        color: black;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: black;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        color: black;
    }
    
    .diagnosis-info {
        background: #f8f9fa;
        padding: 1rem;
        color: black;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        color: black;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
</style>
""", unsafe_allow_html=True)

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
    try:
        model = HybridNet()
        model.load_state_dict(torch.load("hybrid_model.pth", map_location="cpu"))
        model.eval()
        with open("encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# -----------------------------
# Diagnosis Information Database
# -----------------------------
diagnosis_info = {
    "akiec": {
        "name": "Actinic Keratosis & Intraepithelial Carcinoma",
        "description": "Pre-cancerous skin lesions caused by sun damage",
        "severity": "Medium",
        "color": "#ff9800",
        "recommendation": "Consult dermatologist for treatment options"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "Most common type of skin cancer, typically slow-growing",
        "severity": "High",
        "color": "#f44336",
        "recommendation": "Immediate dermatologist consultation required"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "Non-cancerous skin growth, usually harmless",
        "severity": "Low",
        "color": "#4caf50",
        "recommendation": "Monitor for changes, routine checkup"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign skin tumor, usually small and firm",
        "severity": "Low",
        "color": "#4caf50",
        "recommendation": "Generally no treatment needed"
    },
    "mel": {
        "name": "Melanoma",
        "description": "Most dangerous type of skin cancer",
        "severity": "Critical",
        "color": "#9c27b0",
        "recommendation": "URGENT: Immediate medical attention required"
    },
    "nv": {
        "name": "Melanocytic Nevus",
        "description": "Common mole, usually benign",
        "severity": "Low",
        "color": "#4caf50",
        "recommendation": "Monitor for changes in size, color, or shape"
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Blood vessel-related skin lesion",
        "severity": "Low",
        "color": "#2196f3",
        "recommendation": "Usually benign, consult if concerned"
    }
}

# -----------------------------
# Utility Functions
# -----------------------------
def create_confidence_chart(probs, classes):
    """Create a confidence chart for all predictions"""
    fig = go.Figure(data=[
        go.Bar(
            y=classes,
            x=probs * 100,
            orientation='h',
            marker_color=['#667eea' if i == np.argmax(probs) else '#cccccc' for i in range(len(probs))]
        )
    ])
    fig.update_layout(
        title="Confidence Scores for All Diagnoses",
        xaxis_title="Confidence (%)",
        yaxis_title="Diagnosis",
        height=400,
        template="plotly_white"
    )
    return fig

def get_severity_color(severity):
    """Get color based on severity level"""
    colors = {
        "Low": "#4caf50",
        "Medium": "#ff9800", 
        "High": "#f44336",
        "Critical": "#9c27b0"
    }
    return colors.get(severity, "#666666")

# -----------------------------
# Initialize session state
# -----------------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <div class="floating-objects">
        <div class="medical-object">🩺</div>
        <div class="medical-object">🔬</div>
        <div class="medical-object">🧬</div>
        <div class="medical-object">💊</div>
        <div class="medical-object">⚕️</div>
        <div class="medical-object">🏥</div>
    </div>
    <h1>🧬 AI Dermatology Assistant</h1>
    <p>Advanced skin lesion analysis using hybrid AI technology</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📋 About This Tool")
    st.markdown("""
    This AI-powered dermatology assistant uses a hybrid neural network 
    that combines:
    - **Computer Vision** for image analysis
    - **Tabular Data Processing** for patient metadata
    - **HAM10000 Dataset** training for accurate diagnosis
    """)
    
    st.header("⚠️ Important Disclaimer")
    st.warning("""
    This tool is for educational purposes only and should NOT replace 
    professional medical advice. Always consult with a qualified 
    dermatologist for proper diagnosis and treatment.
    """)
    
    st.header("📊 Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "85.2%")
    with col2:
        st.metric("F1-Score", "0.83")
    
    if st.session_state.prediction_history:
        st.header("📈 Recent Predictions")
        for i, pred in enumerate(st.session_state.prediction_history[-3:]):
            st.write(f"**{i+1}.** {pred['diagnosis']} ({pred['confidence']:.1f}%)")

# -----------------------------
# Main Content
# -----------------------------
# Load model
model, encoders = load_model_and_encoders()

if model is None or encoders is None:
    st.error("❌ Could not load the model. Please check if model files exist.")
    st.stop()

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🔍 Diagnosis", "📊 Analytics", "ℹ️ Information"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📷 Image Upload</h3>
            <p>Upload a clear image of the skin lesion for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_image = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"],
            help="Upload a high-quality image of the skin lesion"
        )
        
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Image statistics
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Image Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            img_array = np.array(img)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Width", f"{img.width} px")
            with col_b:
                st.metric("Height", f"{img.height} px")
            with col_c:
                st.metric("Channels", img_array.shape[2])
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>👤 Patient Information</h3>
            <p>Enter patient details for more accurate diagnosis</p>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.slider("📅 Age", 0, 100, 30, help="Patient's age in years")
        sex = st.selectbox("⚧️ Sex", ["male", "female", "unknown"], help="Patient's biological sex")
        site = st.selectbox(
            "📍 Body Site", 
            encoders['anatom_site_general_challenge'].classes_,
            help="Location of the lesion on the body"
        )
        
        # Risk factors
        st.markdown("### 🔍 Risk Assessment")
        sun_exposure = st.radio("☀️ Sun Exposure", ["Low", "Medium", "High"])
        family_history = st.checkbox("👨‍👩‍👧‍👦 Family History of Skin Cancer")
        
        # Predict Button
        predict_button = st.button("🔍 Analyze Lesion", type="primary", use_container_width=True)

    # Prediction Results
    if uploaded_image and predict_button:
        with st.spinner("🔄 Analyzing image..."):
            # Transform image
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
                all_probs = probs[0].numpy()

        # Display results
        st.markdown("""
        <div class="prediction-card">
            <h2>🎯 Diagnosis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        diagnosis = diagnosis_info.get(pred_class, {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Diagnosis</h3>
                <h2>{diagnosis.get('name', pred_class.upper())}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Confidence</h3>
                <h2>{confidence*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            severity = diagnosis.get('severity', 'Unknown')
            color = get_severity_color(severity)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Severity Level</h3>
                <h2 style="color: {color}">{severity}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed diagnosis information
        if diagnosis:
            st.markdown(f"""
            <div class="diagnosis-info">
                <h4>📋 Diagnosis Details</h4>
                <p><strong>Description:</strong> {diagnosis['description']}</p>
                <p><strong>Recommendation:</strong> {diagnosis['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence chart
        all_classes = [encoders['diagnosis'].inverse_transform([i])[0] for i in range(len(all_probs))]
        fig = create_confidence_chart(all_probs, all_classes)
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        risk_score = 0
        if sun_exposure == "High":
            risk_score += 2
        elif sun_exposure == "Medium":
            risk_score += 1
        if family_history:
            risk_score += 2
        if age > 50:
            risk_score += 1
        if pred_class in ["mel", "bcc"]:
            risk_score += 3
        
        risk_level = "Low" if risk_score <= 2 else "Medium" if risk_score <= 4 else "High"
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ Risk Assessment: {risk_level}</h4>
            <p>Based on the analysis and provided information, this case shows {risk_level.lower()} risk factors.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Save to history
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'diagnosis': diagnosis.get('name', pred_class.upper()),
            'confidence': confidence * 100,
            'severity': severity,
            'risk_level': risk_level
        })

with tab2:
    st.header("📊 Analytics Dashboard")
    
    if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = df['severity'].value_counts()
            fig_severity = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Severity Distribution"
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            # Confidence trend
            fig_confidence = px.line(
                df,
                x='timestamp',
                y='confidence',
                title="Confidence Trend Over Time",
                markers=True
            )
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Recent predictions table
        st.subheader("📋 Recent Predictions")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="skin_lesion_analysis_results.csv",
            mime="text/csv"
        )
    else:
        st.info("📈 No predictions yet. Upload an image in the Diagnosis tab to start analyzing!")

with tab3:
    st.header("ℹ️ Skin Lesion Information")
    
    for diagnosis_key, info in diagnosis_info.items():
        with st.expander(f"{info['name']} ({diagnosis_key.upper()})"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Recommendation:** {info['recommendation']}")
            with col2:
                severity_color = get_severity_color(info['severity'])
                st.markdown(f"""
                <div style="background-color: {severity_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                    <strong>{info['severity']} Risk</strong>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🏥 When to See a Doctor
    
    **Immediate attention required if:**
    - Lesion is bleeding or won't heal
    - Rapid changes in size, color, or shape
    - Asymmetry or irregular borders
    - Multiple colors within the lesion
    - Diameter larger than 6mm
    
    **Regular monitoring for:**
    - New moles or lesions
    - Changes in existing moles
    - Family history of skin cancer
    - High sun exposure history
    """)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: grey; padding: 20px;">
    <p>🧬 AI Dermatology Assistant | Powered by Hybrid Neural Networks</p>
    <p><small>This tool is for educational purposes only. Always consult healthcare professionals for medical advice.</small></p>
</div>
""", unsafe_allow_html=True)