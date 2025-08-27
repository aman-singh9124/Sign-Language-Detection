import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import datetime
import cv2
import tempfile
import os

# Load the model
@st.cache_resource
def load_asl_model():
    return load_model("sign_language_model.h5")

model = load_asl_model()
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # A–Z classes

# Helper: Get IST Hour

def get_ist_hour():
    utc_now = datetime.datetime.utcnow()
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    return ist_now.hour

# Check time window: 6PM to 10PM IST
def is_within_allowed_time():
    hour = get_ist_hour()
    return 18 <= hour < 22

# Prediction on image
def predict_image(img):
    img = img.resize((64, 64))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    return classes[np.argmax(pred)]

# Webcam capture function
def capture_webcam_frame():
    cap = cv2.VideoCapture(0)
    st.info("Webcam starting... press 's' to capture or 'q' to quit")

    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        cv2.imshow("Webcam - Press 's' to capture, 'q' to quit", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            captured_image = frame
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Streamlit GUI
st.title("🧠 Sign Language Detection")
st.markdown("Predict ASL signs from uploaded images or webcam input.")

# Time Check
if not is_within_allowed_time():
    st.warning("⛔ Access is only allowed between **6PM to 10PM IST**.")
    st.stop()

# Tabs for UI
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Webcam Detection"])

# --- Tab 1: Image Upload ---
with tab1:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            label = predict_image(image)
            st.success(f"📢 Predicted Sign: **{label}**")

# --- Tab 2: Webcam ---
with tab2:
    st.write("Click the button below to capture from webcam.")
    if st.button("Capture from Webcam"):
        frame = capture_webcam_frame()
        if frame is not None:
            st.image(frame, caption="Captured Frame", channels="BGR", use_column_width=True)
            frame_resized = cv2.resize(frame, (64, 64))
            frame_resized = frame_resized / 255.0
            frame_expanded = np.expand_dims(frame_resized, axis=0)
            pred = model.predict(frame_expanded)
            label = classes[np.argmax(pred)]
            st.success(f"🎥 Real-Time Sign: **{label}**")
