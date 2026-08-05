import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI Polish
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Model Loader with Caching
@st.cache_resource
def load_model_files():
    return load_model("vgg16_brain_tumor_best.h5")

model = load_model_files()

# 3. Sidebar Navigation & Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=90)
    st.title("About Diagnostic System")
    st.caption("Deep Learning Computer Vision Tool")
    
    st.info(
        "This application uses a fine-tuned **VGG16 Neural Network** "
        "to assist in analyzing Brain MRI scans and detecting primary tumor types."
    )
    
    st.divider()
    st.markdown("### 🏷️ Target Classes")
    st.markdown("""
    * **Glioma**: Glial cell tumor
    * **Meningioma**: Meninges tissue tumor
    * **Pituitary**: Pituitary gland tumor
    * **No Tumor**: Healthy brain MRI scan
    """)
    st.divider()
    st.caption("⚠️ *For educational/demonstration purposes only.*")

# 4. Main Header
st.title("🧠 Brain Tumor MRI Classification")
st.markdown("Upload a brain MRI scan below to generate instant CNN-based diagnostic probabilities.")
st.divider()

# 5. File Upload Interface
uploaded_file = st.file_uploader(
    "Choose a Brain MRI Image...", 
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Responsive Column Layout
    col1, col2 = st.columns([1, 1.2], gap="large")

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("🖼️ Input MRI Scan")
        st.image(image, use_container_width=True, caption="Uploaded Image Preview")

    # Image Preprocessing Flow
    img = image.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    with st.spinner("Analyzing scan patterns with VGG16... ⏳"):
        prediction = model.predict(img)

    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    scores = prediction[0]
    
    # Classification Logic
    top_index = np.argmax(scores)
    top_class = class_names[top_index]
    top_score = scores[top_index] * 100

    with col2:
        st.subheader("📊 Diagnostic Summary")
        
        # Primary Prediction Metric Display
        if top_class == "No Tumor":
            st.success(f"### Primary Finding: **{top_class}**")
        else:
            st.error(f"### Primary Finding: **{top_class} Tumor**")

        st.metric(
            label="Model Confidence Level", 
            value=f"{top_score:.2f}%"
        )
        
        st.divider()
        st.markdown("#### Probability Distribution Across All Classes")

        # Class Probability Bars
        df_probs = pd.DataFrame({
            "Class": class_names,
            "Probability (%)": [f"{p*100:.2f}%" for p in scores]
        })
        
        for name, prob in zip(class_names, scores):
            percent = prob * 100
            st.write(f"**{name}**: `{percent:.2f}%`")
            st.progress(float(prob))

        st.divider()
        
        # Bonus Feature: Exportable Diagnosis Summary
        report_text = f"BRAIN MRI DIAGNOSTIC REPORT\n" \
                      f"----------------------------\n" \
                      f"Primary Prediction: {top_class}\n" \
                      f"Confidence: {top_score:.2f}%\n\n" \
                      f"Full Distribution:\n" + \
                      "\n".join([f"- {n}: {p*100:.2f}%" for n, p in zip(class_names, scores)])
        
        st.download_button(
            label="📥 Download Diagnostic Report (.txt)",
            data=report_text,
            file_name="mri_analysis_report.txt",
            mime="text/plain"
        )

else:
    # Clean Initial State Banner
    st.info("👆 Please upload a Brain MRI image above to begin automatic diagnosis.")