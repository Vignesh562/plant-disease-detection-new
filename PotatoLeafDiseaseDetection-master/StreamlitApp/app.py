import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from Home import home
from About import about
from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from datetime import datetime
import os

# --------- Insecure: Appwrite credentials directly in code ---------
APPWRITE_API_ENDPOINT = "https://fra.cloud.appwrite.io/v1"
APPWRITE_PROJECT_ID = "688a1b610038ca502d2f"
APPWRITE_DATABASE_ID = "688a1e470000b53815e8"
APPWRITE_COLLECTION_ID = "688a1ed30009b55657c9"
# ------------------------------------------------------------------

client = Client()
client.set_endpoint(APPWRITE_API_ENDPOINT)
client.set_project(APPWRITE_PROJECT_ID)
account = Account(client)
db = Databases(client)

# --- Authentication ---
if "login" not in st.session_state:
    st.session_state["login"] = False

if not st.session_state["login"]:
    st.title("Login to continue")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            session = account.create_email_session(email, password)
            st.session_state["login"] = True
            st.session_state["user_id"] = session["userId"]
            st.success("Logged in.")
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
    st.stop()

# --- Model and Helper Initialization ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # Custom disease model
mobilenet_model = MobileNetV2(weights="imagenet")
class_names = ["Early Blight", "Late Blight", "Healthy"]

PLANT_KEYWORDS = ["plant", "leaf", "tree", "flower", "maize", "corn", "potato"]

def is_plant_image(img):
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = mobilenet_model.predict(img_array, verbose=0)
    decoded = decode_predictions(preds, top=5)[0]
    labels = [label.lower() for (_, label, _) in decoded]
    return any(any(keyword in label for keyword in PLANT_KEYWORDS) for label in labels)

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels.")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def save_prediction(pred_class, confidence):
    try:
        db.create_document(
            database_id=APPWRITE_DATABASE_ID,
            collection_id=APPWRITE_COLLECTION_ID,
            document_id='unique()',
            data={
                "user": st.session_state["user_id"],
                "prediction": pred_class,
                "confidence": confidence,
                "timestamp": str(datetime.now()),
            }
        )
        st.success("Result saved!")
        del st.session_state["prediction"]
        del st.session_state["confidence"]
    except Exception as e:
        st.warning(f"Failed to save: {str(e)}")

def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the image. Please upload a valid image file.")
            return

        st.image(img, caption="Uploaded Image", use_container_width=True)

        if not is_plant_image(img):
            st.error("❌ This doesn't look like a valid plant leaf. Please upload a clear potato leaf image.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            pred_class = class_names[class_idx]
            confidence = float(predictions[0][class_idx]) * 100

            st.session_state["prediction"] = pred_class
            st.session_state["confidence"] = confidence

            st.subheader("Prediction Results")
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error("⚠️ Error: " + str(e))

def camera():
    camera_image = st.camera_input("Capture a potato leaf image")
    if camera_image is not None:
        try:
            img = Image.open(camera_image).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the captured image. Please try again.")
            return

        st.image(img, caption="Captured Image", use_container_width=True)

        if not is_plant_image(img):
            st.error("❌ This doesn't look like a valid plant leaf. Please upload a clear potato leaf image.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            pred_class = class_names[class_idx]
            confidence = float(predictions[0][class_idx]) * 100

            st.session_state["prediction"] = pred_class
            st.session_state["confidence"] = confidence

            st.subheader("Prediction Results")
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        except Exception as e:
            st.error("⚠️ Error: " + str(e))

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()

    st.sidebar.header("Options")
    option = st.sidebar.selectbox(
        "Choose Your Work",
        ["Upload Image", "Use Camera", "About"],
        index=None
    )

    if option == "Upload Image":
        upload()
    elif option == "Use Camera":
        camera()
    elif option == "About":
        about()

    if "prediction" in st.session_state and "confidence" in st.session_state:
        st.markdown("---")
        if st.button("Save prediction"):
            save_prediction(st.session_state["prediction"], st.session_state["confidence"])

    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.write("Made with ❤️ by Vignesh Parmar")
