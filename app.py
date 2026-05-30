import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import gdown
import os
from gtts import gTTS
import tempfile
from PIL import Image

# ── Page config ──────────────────────────────────────
st.set_page_config(page_title="BrailleVision", page_icon="⠃", layout="centered")

# ── Download model from Google Drive ─────────────────
@st.cache_resource
def load_resources():
    if not os.path.exists("braille_cnn.h5"):
        gdown.download("https://drive.google.com/uc?id=1C70p3IFSWJavG3OzrW9QqLfZw7Dk0aLn", "braille_cnn.h5", quiet=False)

    if not os.path.exists("reverse_map.json"):
        gdown.download("https://drive.google.com/uc?id=14lThgYU0hcihFwSDwMQPZir5IklGAEQx", "reverse_map.json", quiet=False)

    model = load_model("braille_cnn.h5")

    with open("reverse_map.json") as f:
        reverse_map = json.load(f)

    return model, reverse_map

model, reverse_map = load_resources()

# ── OpenCV preprocessing ──────────────────────────────
def preprocess_for_cnn(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32))
    normalized = resized / 255.0
    ready = normalized.reshape(1, 32, 32, 1)
    return ready

# ── Predict ───────────────────────────────────────────
def predict_braille(image_array):
    ready = preprocess_for_cnn(image_array)
    prediction = model.predict(ready)
    class_index = str(np.argmax(prediction))
    letter = reverse_map[class_index]
    confidence = np.max(prediction) * 100
    return letter, confidence

# ── Text to speech ────────────────────────────────────
def speak(text):
    tts = gTTS(text=text, lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# ── UI ────────────────────────────────────────────────
st.title("⠃ BrailleVision")
st.caption("Physical Braille → English using Camera AI")
st.divider()

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Camera"])

with tab1:
    uploaded = st.file_uploader("Upload a Braille image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Recognizing..."):
            letter, confidence = predict_braille(img_array)

        st.success(f"Predicted Letter:  {letter.upper()}")
        st.metric("Confidence", f"{confidence:.1f}%")

        if st.button("🔊 Read aloud"):
            audio = speak(f"The Braille letter is {letter}")
            st.audio(audio)

with tab2:
    cam_img = st.camera_input("Point camera at Braille")

    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)

        with st.spinner("Recognizing..."):
            letter, confidence = predict_braille(img_array)

        st.success(f"Predicted Letter:  {letter.upper()}")
        st.metric("Confidence", f"{confidence:.1f}%")

        if st.button("🔊 Read aloud", key="cam_speak"):
            audio = speak(f"The Braille letter is {letter}")
            st.audio(audio)
