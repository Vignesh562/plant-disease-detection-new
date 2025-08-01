import streamlit as st
import numpy as np
import os
import pandas as pd
from datetime import datetime
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import wikipedia

from Home import home
from About import about

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = ["Early Blight", "Late Blight", "Healthy"]

wikipedia.set_lang("en")

def fetch_wiki_summary(title):
    try:
        return wikipedia.summary(title, sentences=2)
    except Exception:
        # Fallback hardcoded description for known diseases
        fallback_descriptions = {
            "early blight": "Early blight is a common fungal disease affecting potatoes, characterized by dark spots on leaves and fruit, leading to reduced yield.",
            "late blight": "Late blight is a serious potato disease caused by Phytophthora infestans, resulting in rapid decay of leaves and tubers, often causing significant crop loss."
        }
        return fallback_descriptions.get(title.lower(), "No Wikipedia summary found.")

# Disease info dict with fallback in place
disease_info = {
    "Early Blight": {
        "description": fetch_wiki_summary("Early blight"),
        "treatment": "Use fungicides like chlorothalonil or mancozeb. Practice crop rotation and remove infected plant debris."
    },
    "Late Blight": {
        "description": fetch_wiki_summary("Late blight"),
        "treatment": "Apply fungicides like metalaxyl or cymoxanil. Remove and destroy infected plants. Avoid overhead irrigation."
    },
    "Healthy": {
        "description": "No signs of disease detected on the potato leaf.",
        "treatment": "No treatment necessary. Maintain regular monitoring to catch early signs of disease."
    }
}

def log_prediction(image_name, prediction, confidence):
    df = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), image_name, prediction, f"{confidence:.2f}%"]],
                      columns=["Timestamp", "Image", "Prediction", "Confidence"])
    if not os.path.exists(HISTORY_PATH):
        df.to_csv(HISTORY_PATH, index=False)
    else:
        df.to_csv(HISTORY_PATH, mode='a', header=False, index=False)

def show_history():
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        search = st.text_input("🔍 Search by disease name")
        if search:
            df = df[df['Prediction'].str.contains(search, case=False, na=False)]
        st.dataframe(df[::-1], use_container_width=True)

        st.markdown("### 📊 Prediction Statistics")
        stats = df['Prediction'].value_counts()
        st.bar_chart(stats)
    else:
        st.info("No prediction history available yet.")

def preprocess_image(img: Image.Image):
    img = img.resize((256, 256))
    img_array = np.array(img)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels.")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def upload():
    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)

    if dark_mode:
        bg_style = "#121212"
        text_color = "#e0e0e0"
        desc_bg = "#1e1e1e"
        pred_text = "#a5d6a7"
    else:
        bg_style = "linear-gradient(to right, #f0f9ff, #e0f7fa)"
        text_color = "#2E7D32"
        desc_bg = "#ffffff"
        pred_text = "#2e7d32"

    st.markdown(f"""
        <style>
            .reportview-container .main {{
                background: {bg_style};
                font-family: 'Segoe UI', sans-serif;
                color: {text_color};
            }}
            h2, h3, h4 {{
                color: {text_color};
            }}
            .stButton>button {{
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                transition: background 0.3s ease;
            }}
            .stButton>button:hover {{
                background-color: #45a049;
            }}
        </style>
        <h2 style='text-align: center;'>🌿 Upload a Potato Leaf Image</h2>
        <p style='text-align: center; color: #555;'>AI-powered leaf disease detection and treatment suggestions.</p>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the image. Please upload a valid image file.")
            return

        st.image(img, caption="Uploaded Image", use_container_width=True)

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction(uploaded_file.name, disease, confidence)

            st.markdown(f"""
                <div style='padding: 1.2rem; background: {desc_bg}; color: {pred_text}; border-left: 6px solid #66bb6a; border-radius: 8px; box-shadow: 2px 2px 5px #ccc;'>
                    <h3>🧪 Prediction: {disease}</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                <div style='margin-top: 1.5em; background: {desc_bg}; padding: 1rem; border-radius: 8px; box-shadow: 1px 1px 3px #aaa;'>
                    <h4>📖 Disease Description</h4>
                    <p>{disease_info[disease]['description']}</p>
                    <h4>💊 Treatment Suggestions</h4>
                    <p>{disease_info[disease]['treatment']}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Something went wrong while processing the image. Please try a different image.\nError: {e}")

def camera():
    st.header("📸 Capture a Potato Leaf Image")

    camera_image = st.camera_input("")

    if camera_image is not None:
        try:
            img = Image.open(camera_image).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the captured image. Please try again.")
            return

        st.image(img, caption="Captured Image", use_container_width=True)

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction("Camera Capture", disease, confidence)

            dark_mode = st.sidebar.checkbox("🌙 Dark Mode", value=False)
            if dark_mode:
                desc_bg = "#1e1e1e"
                pred_text = "#a5d6a7"
            else:
                desc_bg = "#ffffff"
                pred_text = "#2e7d32"

            st.markdown(f"""
                <div style='padding: 1.2rem; background: {desc_bg}; color: {pred_text}; border-left: 6px solid #66bb6a; border-radius: 8px; box-shadow: 2px 2px 5px #ccc;'>
                    <h3>🧪 Prediction: {disease}</h3>
                    <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                </div>
                <div style='margin-top: 1.5em; background: {desc_bg}; padding: 1rem; border-radius: 8px; box-shadow: 1px 1px 3px #aaa;'>
                    <h4>📖 Disease Description</h4>
                    <p>{disease_info[disease]['description']}</p>
                    <h4>💊 Treatment Suggestions</h4>
                    <p>{disease_info[disease]['treatment']}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Something went wrong while processing the image. Please try a different image.\nError: {e}")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()

    with st.sidebar:
        st.markdown("""
            <style>
                .sidebar .sidebar-content {padding: 1rem;}
            </style>
            <h3 style='color:#2e7d32;'>🌿 Navigation</h3>
            """, unsafe_allow_html=True)

        option = st.selectbox("Choose Your Work", ["Upload Image", "Use Camera", "View History", "About"])

    if option == "Upload Image":
        upload()
    elif option == "Use Camera":
        camera()
    elif option == "View History":
        show_history()
    elif option == "About":
        about()

    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.markdown("<p style='text-align:center;'>Made with ❤️ by Vignesh, Pankaj, Denial</p>", unsafe_allow_html=True)
