import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from Home import home import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from PIL import Image, UnidentifiedImageError
from Home import home
from About import about
import os 

# ✅ Appwrite imports
from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.id import ID
from appwrite.exception import AppwriteException

# ✅ Appwrite credentials – replace with your actual project info
APPWRITE_ENDPOINT = "https://fra.cloud.appwrite.io/v1"
APPWRITE_PROJECT_ID = "688a1b610038ca502d2f"

# ✅ Appwrite client setup
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT).set_project(APPWRITE_PROJECT_ID)
account = Account(client)

# ✅ Initialize session state for user
if "user" not in st.session_state:
    st.session_state.user = None

# ✅ Login/Signup UI
def login_signup_ui():
    st.sidebar.header("🔐 Login / Signup")
    mode = st.sidebar.radio("Mode", ["Login", "Signup"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button(mode):
        if not email or not password:
            st.warning("Please enter both email and password.")
            return

        try:
            if mode == "Signup":
                account.create(user_id=ID.unique(), email=email, password=password)
                st.success("✅ Signup successful. Please login.")
            else:
                session = account.create_email_session(email=email, password=password)
                if "$id" in session:
                    client.set_session(session["$id"])
                    user_info = account.get()
                    st.session_state.user = user_info
                    st.success(f"✅ Logged in as {user_info['email']}")
                else:
                    st.error("Login failed. No session ID.")
        except AppwriteException as e:
            st.error(f"❌ {e.message}")

    if st.session_state.user:
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.experimental_rerun()

# ✅ Show login/signup first
login_signup_ui()
if not st.session_state.user:
    st.stop()

# -----------------------------------------------
# 🍃 Original Potato Leaf Detection Logic Below
# -----------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "potatoes.h5")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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

def upload():
    uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the image.")
            return

        st.image(img, caption="Uploaded Image", use_container_width=True)

        if not is_plant_image(img):
            st.error("❌ Not a valid plant leaf image.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            st.subheader("Prediction Results")
            st.success(f"Prediction: {class_names[class_idx]}")
            st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}%")
        except Exception as e:
            st.error("⚠️ Error while processing the image.")

def camera():
    camera_image = st.camera_input("Capture a potato leaf image")
    if camera_image is not None:
        try:
            img = Image.open(camera_image).convert("RGB")
        except UnidentifiedImageError:
            st.error("❌ Unable to read the captured image.")
            return

        st.image(img, caption="Captured Image", use_container_width=True)

        if not is_plant_image(img):
            st.error("❌ Not a valid plant leaf image.")
            return

        try:
            img_array = preprocess_image(img)
            predictions = model.predict(img_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            st.subheader("Prediction Results")
            st.success(f"Prediction: {class_names[class_idx]}")
            st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}%")
        except Exception as e:
            st.error("⚠️ Error while processing the captured image.")

def preprocess_image(img):
    img = img.resize((256, 256))
    img_array = np.array(img)
    if img_array.ndim != 3 or img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 color channels.")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# 🧭 Page Routing
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

            st.subheader("Prediction Results")
            st.success(f"Prediction: {class_names[class_idx]}")
            st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}%")
        except Exception as e:
            st.error("⚠️ Something went wrong while processing the image. Please try a different image.")

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

            st.subheader("Prediction Results")
            st.success(f"Prediction: {class_names[class_idx]}")
            st.info(f"Confidence: {predictions[0][class_idx]*100:.2f}%")
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
