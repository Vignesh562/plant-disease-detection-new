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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")

# Load the trained custom CNN model without compiling
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Load MobileNetV2 for plant/leaf validation
mobilenet_model = MobileNetV2(weights="imagenet")

class_names = ["Early Blight", "Late Blight", "Healthy"]

# Check if image is likely a plant using MobileNetV2
PLANT_KEYWORDS = ["plant", "leaf", "tree", "flower", "maize", "corn", "potato"]
# Treatment and description data
disease_info = {
    "Early Blight": {
        "description": "Early blight is a common potato disease caused by the fungus *Alternaria solani*. It causes dark spots on older leaves.",
        "treatment": "Use fungicides like chlorothalonil or mancozeb. Practice crop rotation and remove infected plant debris."
    },
    "Late Blight": {
        "description": "Late blight is a devastating disease caused by the oomycete *Phytophthora infestans*. It appears as water-soaked lesions on leaves and stems.",
        "treatment": "Apply fungicides like metalaxyl or cymoxanil. Remove and destroy infected plants. Avoid overhead irrigation."
    },
    "Healthy": {
        "description": "No signs of disease detected on the potato leaf.",
        "treatment": "No treatment necessary. Maintain regular monitoring to catch early signs of disease."
    }
}

# Keywords to check in predictions
PLANT_KEYWORDS = [
    "plant", "leaf", "tree", "flower", "maize", "corn", "potato",
    "leaves", "foliage", "green", "branch", "bush", "grass", "crop", "vegetable", "shrub"
]

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


def upload():
    st.markdown("""
    <h2 style='text-align: center;'>🌿 Upload a Potato Leaf Image</h2>
    <p style='text-align: center; color: grey;'>This tool detects potato leaf diseases using deep learning.</p>
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
            st.warning("⚠️ This doesn't appear to be a valid plant or leaf image. Please try a different one.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])

            disease = class_names[class_idx]

            st.markdown("""
            <div style='padding: 1rem; background-color: #e8f5e9; border-radius: 10px;'>
                <h4>🧪 Prediction: <span style='color: #2e7d32;'>%s</span></h4>
                <p><strong>Confidence:</strong> %.2f%%</p>
            </div>
            """ % (disease, predictions[0][class_idx]*100), unsafe_allow_html=True)

            st.markdown("""
            <h4>📖 Disease Description</h4>
            <p>%s</p>
            <h4>💊 Treatment Suggestions</h4>
            <p>%s</p>
            """ % (disease_info[disease]["description"], disease_info[disease]["treatment"]), unsafe_allow_html=True)

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
            st.warning("⚠️ This doesn't appear to be a valid plant or leaf image. Please try again.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])

            disease = class_names[class_idx]

            st.markdown("""
            <div style='padding: 1rem; background-color: #e8f5e9; border-radius: 10px;'>
                <h4>🧪 Prediction: <span style='color: #2e7d32;'>%s</span></h4>
                <p><strong>Confidence:</strong> %.2f%%</p>
            </div>
            """ % (disease, predictions[0][class_idx]*100), unsafe_allow_html=True)

            st.markdown("""
            <h4>📖 Disease Description</h4>
            <p>%s</p>
            <h4>💊 Treatment Suggestions</h4>
            <p>%s</p>
            """ % (disease_info[disease]["description"], disease_info[disease]["treatment"]), unsafe_allow_html=True)

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
        option = st.selectbox("Choose Your Work", ["Upload Image", "Use Camera", "About"], index=None)

    if option == "Upload Image":
        upload()
    elif option == "Use Camera":
        camera()
    elif option == "About":
        about()

    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.markdown("<p style='text-align:center;'>Made with ❤️ by Vignesh Parmar</p>", unsafe_allow_html=True)
