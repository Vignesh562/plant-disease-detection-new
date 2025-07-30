import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from Home import home
from About import about
import os
import pandas as pd
from datetime import datetime
import wikipedia
from appwrite_config import client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from appwrite.id import ID
from appwrite.exception import AppwriteException

# Appwrite setup
account = Account(client)
databases = Databases(client)

# Session state for user
if "user" not in st.session_state:
    st.session_state.user = None

def login_signup_ui():
    st.sidebar.header("\U0001F510 Login / Signup")
    mode = st.sidebar.radio("Choose Mode", ["Login", "Signup"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(mode):
        try:
            if mode == "Signup":
                account.create(user_id=ID.unique(), email=email, password=password)
                st.success("\u2705 Signup successful. Please login.")
            else:
                session = account.create_email_session(email=email, password=password)
                user_info = account.get()
                st.session_state.user = user_info
                st.success(f"\u2705 Logged in as {user_info['email']}")
        except AppwriteException as e:
            st.error(f"\u274C {e.message}")

login_signup_ui()

if not st.session_state.user:
    st.stop()

# Paths and models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")
HISTORY_PATH = os.path.join(BASE_DIR, "prediction_history.csv")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = ["Early Blight", "Late Blight", "Healthy"]

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

def log_prediction(user_email, image_name, prediction, confidence):
    data = {
        "user_email": user_email,
        "image_name": image_name,
        "predicted_disease": prediction,
        "confidence": f"{confidence:.2f}%",
        "timestamp": datetime.now().isoformat()
    }
    try:
        databases.create_document(
            database_id="your-db-id",
            collection_id="predictions",
            document_id=ID.unique(),
            data=data
        )
    except AppwriteException as e:
        st.warning(f"\ud83d\udcc2 Appwrite DB Error: {e.message}")

    df = pd.DataFrame([[data['timestamp'], image_name, prediction, data['confidence']]],
                      columns=["Timestamp", "Image", "Prediction", "Confidence"])
    if not os.path.exists(HISTORY_PATH):
        df.to_csv(HISTORY_PATH, index=False)
    else:
        df.to_csv(HISTORY_PATH, mode='a', header=False, index=False)

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels.")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def show_history():
    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        search = st.text_input("\U0001F50D Search by disease name")
        if search:
            df = df[df['Prediction'].str.contains(search, case=False, na=False)]
        st.dataframe(df[::-1], use_container_width=True)

        st.markdown("### \U0001F4CA Prediction Statistics")
        stats = df['Prediction'].value_counts()
        st.bar_chart(stats)
    else:
        st.info("No prediction history available yet.")

def upload():
    dark_mode = st.sidebar.toggle("\U0001F319 Dark Mode", value=False)
    bg_style = "#121212" if dark_mode else "linear-gradient(to right, #f0f9ff, #e0f7fa)"
    text_color = "#e0e0e0" if dark_mode else "#2E7D32"
    desc_bg = "#1e1e1e" if dark_mode else "#ffffff"

    st.markdown(f"""
    <style>
        .reportview-container .main {{ background: {bg_style}; color: {text_color}; }}
        h2, h3, h4 {{ color: {text_color}; }}
    </style>
    <h2 style='text-align: center;'>\U0001F33F Upload a Potato Leaf Image</h2>
    <p style='text-align: center;'>AI-powered leaf disease detection and treatment suggestions.</p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction(st.session_state.user['email'], uploaded_file.name, disease, confidence)

            st.markdown(f"""
            <div style='padding:1rem;background:{desc_bg};border-radius:8px;'>
                <h3>\U0001F9EA Prediction: {disease}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                <h4>\U0001F4D6 Description</h4>
                <p>{disease_info[disease]['description']}</p>
                <h4>\U0001F48A Treatment</h4>
                <p>{disease_info[disease]['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("\u26A0\uFE0F Failed to process image.")

def camera():
    st.header("\U0001F4F8 Capture a Potato Leaf Image")
    camera_image = st.camera_input("")
    if camera_image is not None:
        try:
            img = Image.open(camera_image).convert("RGB")
            st.image(img, caption="Captured Image", use_container_width=True)
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            disease = class_names[class_idx]
            confidence = predictions[0][class_idx] * 100

            log_prediction(st.session_state.user['email'], "Camera Capture", disease, confidence)

            st.markdown(f"""
            <div style='padding:1rem;background:#fff;border-radius:8px;'>
                <h3>\U0001F9EA Prediction: {disease}</h3>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                <h4>\U0001F4D6 Description</h4>
                <p>{disease_info[disease]['description']}</p>
                <h4>\U0001F48A Treatment</h4>
                <p>{disease_info[disease]['treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error("\u26A0\uFE0F Failed to process captured image.")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()

    with st.sidebar:
        st.markdown("""
        <h3 style='color:#2e7d32;'>\U0001F33F Navigation</h3>
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
    st.info("\ud83d\udccc Navigate using the sidebar.")
    st.markdown("<p style='text-align:center;'>Made with ❤️ by Vignesh Parmar</p>", unsafe_allow_html=True)
