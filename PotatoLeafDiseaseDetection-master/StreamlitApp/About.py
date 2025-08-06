import streamlit as st

def about():
    st.title("About This Project")

    st.write("""
        This project is designed to perform **real-time plant leaf disease detection** 
        using advanced machine learning and computer vision techniques. It incorporates 
        various features and functionalities to enhance the user experience and improve 
        the model's reliability. Below is a detailed explanation of each feature used in this project:
    """)

    st.subheader("Camera Input")
    st.write("""
        - Users can capture an image of a plant's leaf using their webcam.
        - The captured image is preprocessed and passed through three different models for predictions:
            - **Custom CNN Model**: A specialized model trained on the dataset.
        - Predictions from the Custom CNN Model, along with their confidence scores, are displayed for comparison.
    """)

    st.subheader("Managed History & Downloadable Reports")
    st.write("""
        - The system maintains a **comprehensive history** of all your disease detection records.
        - Each prediction is **automatically stored** with details like:
            - Image name
            - Detected disease
            - Confidence score
            - Timestamp
        - Users can **filter history** by disease type or date for quick access.
        - All records are **available to download** as a CSV or PDF for offline usage and analysis.
    """)

    st.subheader("Visual Analytics of Detection Trends")
    st.write("""
        - A **dynamic chart** shows the distribution of detected plant leaf diseases over time.
        - It calculates the **disease occurrence ratio**, highlighting which disease is most frequent.
        - This visual insight helps in identifying trends and making informed farming decisions.
        - Charts like **bar graphs** or **pie charts** are used for intuitive understanding.
    """)

    st.subheader("Bounding Box for Classification Area")
    st.write("""
        - A **bounding box** dynamically highlights the region of the plant leaf where the 
          model focuses its analysis.
        - This bounding box is calculated using the Grad-CAM heatmap and dynamically adjusts 
          to the most activated regions in the image.
    """)

    st.subheader("🛠Advanced Controls")
    st.write("""
        - The video feed includes **Play** and **Stop** buttons for user control.
        - Frames are analyzed in real-time, but after every **15 frames**, the system confirms 
          predictions with a bounding box and the most probable answer for improved reliability.
    """)

    st.subheader("Dashboard-Like Functionality")
    st.write("""
        - The application is designed like a dashboard, providing real-time insights and visualizations.
        - Users can navigate between features such as **Real-Time Detection**, **Capture Image**, 
          and **About Page** with ease.
    """)

    st.subheader(" Technologies Used")
    st.write("""
        - **TensorFlow**: For training and deploying the machine learning models.
        - **OpenCV**: For image and video processing.
        - **Streamlit**: For creating an interactive and user-friendly web application.
        - **Grad-CAM**: For heatmap-based visualizations of model activations.
        - **Custom Trained CNN Models**: Shows use of CNN architecture.
        - **Pre-Trained Models**: Transfer learning from state-of-the-art architectures like Inception and ResNet.
    """)

    st.info("This application demonstrates the power of AI in agriculture by aiding in the early detection of plant leaf diseases, helping farmers make informed decisions.")
