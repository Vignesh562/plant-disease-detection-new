import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from Home import home
from About import about
import time 
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")

# Load the trained custom CNN model
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["Early Blight", "Late Blight", "Healthy"]

# Check if image is likely a valid plant leaf
def is_probably_leaf(image_array):
    if image_array.std() < 5:
        return False
    if image_array.mean() > 240 or image_array.mean() < 15:
        return False
    return True

def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        img_array = preprocess_image(img)

        if not is_probably_leaf(img_array):
            st.error("❌ This doesn't look like a valid plant leaf. Please upload a clear image of a potato leaf.")
            return

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        st.subheader("Prediction Results")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

        st.subheader("Generating Grad-CAM")
        heatmap = grad_cam(model, img_array, "conv2d_116")  # Change to your CNN's last conv layer
        result_img = overlay_heatmap_Img(heatmap, img, opacity=0.4)
        st.image(result_img, caption="Grad-CAM Result", use_container_width=True)

def camera():
    camera_image = st.camera_input("Capture a potato leaf image")
    if camera_image is not None:
        img = Image.open(camera_image)
        st.image(img, caption="Captured Image", use_container_width=True)
        img_array = preprocess_image(img)

        if not is_probably_leaf(img_array):
            st.error("❌ This doesn't look like a valid plant leaf. Please try again with a clear leaf image.")
            return

        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])

        st.subheader("Prediction Results")
        st.success(f"Prediction: {class_names[class_idx]}")
        st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}")

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    gradients = tape.gradient(loss, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = np.zeros(conv_outputs.shape[:-1])

    for i in range(conv_outputs.shape[-1]):
        heatmap += pooled_gradients[i] * conv_outputs[:, :, i]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap

def overlay_heatmap_Img(heatmap, img, opacity):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - opacity, heatmap, opacity, 0)
    return superimposed_img

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
