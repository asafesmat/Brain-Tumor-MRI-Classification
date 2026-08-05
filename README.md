# 🧠 Brain Tumor MRI Classification

![App Screenshot](Images\image.png)

A Deep Learning web application for Brain Tumor MRI Classification using VGG16 Transfer Learning and Streamlit.

---

## 📌 Overview

This project classifies brain MRI images into four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

The model was trained using TensorFlow/Keras with VGG16 as the backbone network.

---

## 🚀 Features

- Upload MRI image
- Automatic preprocessing
- Predict tumor type
- Displays predicted class
- Streamlit web interface

---

## 🛠️ Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Pillow

---

## 📂 Project Structure

```
Brain-Tumor-MRI-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

## ▶️ Run Locally

Create virtual environment

```bash
python -m venv env
```

Activate it

Windows

```bash
env\Scripts\activate
```

Install packages

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📊 Model

Backbone:

- VGG16 (Transfer Learning)

Loss Function:

- Sparse Categorical Crossentropy

Optimizer:

- Adamax

---

## 👨‍💻 Author

Asaf Esmat