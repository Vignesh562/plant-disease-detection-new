import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from Home import home
from About import about
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")

# Load the trained custom CNN model without compiling
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

class_names = ["Early Blight", "Late Blight", "Healthy"]

def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        st.subheader("Prediction Results")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

def camera():
    camera_image = st.camera_input("Capture a potato leaf image")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", use_container_width=True)

        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        st.subheader("Prediction Results")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

def preprocess_image(img):
    img = img.convert("RGB")  # Ensures 3 channels
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        home()

    st.sidebar.header("Options")
    option = st.sidebar.selectbox("Choose Your Work", ["Upload Image", "Use Camera", "About"], index=None)

    if option == "Upload Image":
        upload()
    elif option == "Use Camera":
        camera()
    elif option == "About":
        about()

    st.markdown("---")
    st.info("📌 Navigate to different sections using the sidebar.")
    st.write("Made with ❤️ by Vignesh Parmar")
