# 🧠 Brain Tumor MRI Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project is a **Deep Learning web application** that classifies Brain MRI images into four categories using **Transfer Learning with VGG16**.

The application allows users to upload an MRI image through a **Streamlit interface**, and the trained model predicts the tumor type instantly.

---

## 🎯 Classes

The model classifies MRI images into:

- 🧠 Glioma
- 🧠 Meningioma
- 🧠 Pituitary
- ✅ No Tumor

---

## 🏗️ Model Architecture

- VGG16 (Pretrained on ImageNet)
- Transfer Learning
- Global Average Pooling
- Fully Connected Layers
- Softmax Output Layer

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pillow
- Streamlit

---

## 📂 Project Structure

```
Brain-Tumor-MRI-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── images/
    └── image.png 
```

---

## 🚀 Run Locally

### Clone Repository

```bash
git clone https://github.com/asafesmat/Brain-Tumor-MRI-Classification.git

cd Brain-Tumor-MRI-Classification
```

### Create Virtual Environment

```bash
python -m venv env
```

### Activate Environment

Windows

```bash
env\Scripts\activate
```

Linux / Mac

```bash
source env/bin/activate
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Value |
|---------|-------|
| Accuracy | **87.6%** |
| Loss | 1.12 |
| Backbone | VGG16 |

---

## 🖼️ Application Preview

ضع صورة للتطبيق داخل المجلد:

```
images/app.png
```

ثم ألغِ التعليق عن السطر التالي بعد إضافة الصورة:

```markdown
<!-- ![App Screenshot](images/app.png) -->
```

---

## 📈 Future Improvements

- Deploy on Streamlit Community Cloud
- Grad-CAM Visualization
- Better UI Design
- Confidence Score Visualization
- Support Batch Prediction

---

## 👨‍💻 Author

**Asaf Esmat**

- LinkedIn:[ https://www.linkedin.com/in/asaf-esmat](https://www.linkedin.com/in/asaf-esmat-557004357?utm_source=share_via&utm_content=profile&utm_medium=member_ios)
- GitHub: https://github.com/asafesmat

---

## ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.
