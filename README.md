# 🧠 Skin Lesion Classifier (Hybrid CNN + Tabular AI Model)

A Streamlit-based web app that predicts **skin lesion types** using both image and tabular data (age, sex, and body location).  
Built using a hybrid deep learning model trained on the **HAM10000** dataset.

---

## 🚀 Demo

👉 **Try it live**: [https://skin-lesion-classifier.streamlit.app/](#)

---

## 🩺 What It Does

- Accepts lesion **image upload**
- Accepts patient details:
  - Age
  - Sex (Male/Female)
  - Anatomical site (e.g., torso, back)
- Predicts one of the **7 skin lesion types**:
  - akiec (Actinic keratoses)
  - bcc (Basal cell carcinoma)
  - bkl (Benign keratosis-like lesions)
  - df (Dermatofibroma)
  - melanoma
  - nevus (nv)
  - vasc (Vascular lesions)

---

## 🧠 Model Overview

- **Image Input**: ResNet18 (pretrained)
- **Tabular Input**: Age, Sex, Site (embedded & encoded)
- **Fusion**: Concatenates image & tabular features
- **Output**: Softmax layer with 7-class classification

---

## 📁 Folder Structure

project/
├── streamlit_app.py # 🔥 Main Streamlit app
├── hybrid_model.pth # 🧠 Trained model weights
├── encoders.pkl # 🎛️ Scikit-learn label encoders
├── requirements.txt # 📦 Dependencies
└── test_samples/ # 🧪 Test images + metadata


---

## 💻 How to Run Locally

1. Clone this repo:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run streamlit_app.py
```
