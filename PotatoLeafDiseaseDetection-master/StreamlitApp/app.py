import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from Home import home
from About import about
import os
import pandas as pd
from datetime import datetime
import wikipedia

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
mobilenet_model = MobileNetV2(weights="imagenet")

class_names = ["Early Blight", "Late Blight", "Healthy"]

# Disease info with placeholders
wikipedia.set_lang("en")

def fetch_wiki_summary(title):
    try:
        return wikipedia.summary(title, sentences=2)
    except:
        return "No Wikipedia summary found."

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

PLANT_KEYWORDS = ["plant", "leaf", "tree", "flower", "maize", "corn", "potato"]

# Enhanced plant/leaf image validation with label confidence
def is_plant_image(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = mobilenet_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    labels = [label.lower() for (_, label, _) in decoded]
    if any(any(keyword in label for keyword in PLANT_KEYWORDS) for label in labels):
        return True
    return False

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

        # Statistics
        st.markdown("### 📊 Prediction Statistics")
        stats = df['Prediction'].value_counts()
        st.bar_chart(stats)
    else:
        st.info("No prediction history available yet.")

def upload():
    st.markdown("""
    <h2 style='text-align: center; color: #4CAF50;'>🌿 Upload a Potato Leaf Image</h2>
    <p style='text-align: center; color: #666;'>AI-powered leaf disease detection and treatment suggestions.</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the image. Please upload a valid image file.")
            return

        st.image(img, caption="Uploaded Image", use_container_width=True)

        if not is_plant_image(img):
            st.warning("⚠️ This doesn't seem like a potato plant or leaf. Suggestions: Try maize, tomato, or other crop images.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction(uploaded_file.name, disease, confidence)

            st.markdown(f"""
            <div style='padding: 1rem; background-color: #F1F8E9; border-left: 5px solid #4CAF50;'>
                <h3>🧪 Prediction: <span style='color: #2E7D32;'>{disease}</span></h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            <div style='margin-top: 1em;'>
                <h4>📖 Disease Description</h4>
                <p>{disease_info[disease]['description']}</p>
                <h4>💊 Treatment Suggestions</h4>
                <p>{disease_info[disease]['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("⚠️ Something went wrong while processing the image. Please try a different image.")

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

        if not is_plant_image(img):
            st.warning("⚠️ This doesn't seem like a potato plant or leaf. Suggestions: Try maize, tomato, or other crop images.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction("Camera Capture", disease, confidence)

            st.markdown(f"""
            <div style='padding: 1rem; background-color: #F1F8E9; border-left: 5px solid #4CAF50;'>
                <h3>🧪 Prediction: <span style='color: #2E7D32;'>{disease}</span></h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            <div style='margin-top: 1em;'>
                <h4>📖 Disease Description</h4>
                <p>{disease_info[disease]['description']}</p>
                <h4>💊 Treatment Suggestions</h4>
                <p>{disease_info[disease]['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("⚠️ Something went wrong while processing the image. Please try a different image.")

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels.")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

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
        option = st.selectbox("Choose Your Work", ["Upload Image", "Use Camera", "View History", "About"], index=None)

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
    st.markdown("<p style='text-align:center;'>Made with ❤️ by Vignesh Parmar</p>", unsafe_allow_html=True)
