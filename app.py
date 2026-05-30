import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import json
import gdown
import os
from gtts import gTTS
import tempfile
from PIL import Image

st.set_page_config(page_title="BrailleVision", page_icon="⠃", layout="centered")

@st.cache_resource
def load_resources():
    if not os.path.exists("braille_cnn.onnx"):
        gdown.download("https://drive.google.com/uc?id=1keYm335Ks2M7XVnoqmKTiBGprRb2FlYJ", "braille_cnn.onnx", quiet=False)
    if not os.path.exists("reverse_map.json"):
        gdown.download("https://drive.google.com/uc?id=14lThgYU0hcihFwSDwMQPZir5IklGAEQx", "reverse_map.json", quiet=False)

    session = ort.InferenceSession("braille_cnn.onnx")
    with open("reverse_map.json") as f:
        reverse_map = json.load(f)
    return session, reverse_map

session, reverse_map = load_resources()

# ── Predict single cell ───────────────────────────────
def predict_cell(cell_img):
    # Convert to grayscale
    gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    
    # Invert if background is dark (dots are light)
    if np.mean(gray) < 127:
        gray = cv2.bitwise_not(gray)
    
    # Resize to training size
    resized = cv2.resize(gray, (32, 32))
    normalized = resized / 255.0
    ready = normalized.reshape(1, 32, 32, 1).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: ready})[0]
    class_index = str(np.argmax(prediction))
    letter = reverse_map[class_index]
    confidence = float(np.max(prediction)) * 100
    return letter, confidence

# ── OpenCV: find all Braille cells in image ───────────
def find_and_predict_all(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all dot contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 20 < area < 500:
            cx = x + w // 2
            cy = y + h // 2
            dots.append(cx)

    if not dots:
        return "", [], image_array

    # Find gaps between dots to detect cell boundaries
    dots_x = sorted(set(dots))
    
    # Find large gaps — these are spaces between Braille cells
    gaps = []
    for i in range(len(dots_x) - 1):
        gap = dots_x[i+1] - dots_x[i]
        gaps.append((dots_x[i], dots_x[i+1], gap))
    
    if not gaps:
        return "", [], image_array

    # Average small gap = dot spacing within a cell
    all_gaps = [g[2] for g in gaps]
    avg_gap = np.median(all_gaps)
    
    # A cell boundary is where gap > 1.5x average gap
    cell_boundaries = [0]
    for (x1, x2, gap) in gaps:
        if gap > avg_gap * 1.5:
            boundary = (x1 + x2) // 2
            cell_boundaries.append(boundary)
    cell_boundaries.append(image_array.shape[1])

    if len(cell_boundaries) < 2:
        return "", [], image_array

    result_letters = []
    result_confidences = []
    annotated = image_array.copy()

    for i in range(len(cell_boundaries) - 1):
        x_start = cell_boundaries[i]
        x_end = cell_boundaries[i+1]

        # Skip very narrow strips (noise)
        if x_end - x_start < 10:
            continue

        # Crop full height, cell width
        cell = image_array[:, x_start:x_end]

        if cell.size == 0:
            continue

        letter, confidence = predict_cell(cell)
        result_letters.append(letter)
        result_confidences.append(confidence)

        # Draw box
        cv2.rectangle(annotated, (x_start, 0), (x_end, image_array.shape[0]), (0, 255, 0), 2)
        cv2.putText(annotated, letter.upper(), (x_start + 2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    sentence = "".join(result_letters)
    return sentence, result_confidences, annotated
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

        with st.spinner("Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)

        if sentence:
            st.image(annotated, caption="Detected cells", use_column_width=True)
            st.markdown("### 📝 Recognized Text")
            st.success(sentence.upper())
            if confidences:
                avg = sum(confidences) / len(confidences)
                st.metric("Average Confidence", f"{avg:.1f}%")
            if st.button("🔊 Read aloud"):
                audio = speak(sentence)
                st.audio(audio)
        else:
            st.warning("No Braille cells detected. Try a clearer image.")

with tab2:
    cam_img = st.camera_input("Point camera at Braille")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        img_array = np.array(img)

        with st.spinner("Reading Braille..."):
            sentence, confidences, annotated = find_and_predict_all(img_array)

        if sentence:
            st.image(annotated, caption="Detected cells", use_column_width=True)
            st.markdown("### 📝 Recognized Text")
            st.success(sentence.upper())
            if confidences:
                avg = sum(confidences) / len(confidences)
                st.metric("Average Confidence", f"{avg:.1f}%")
            if st.button("🔊 Read aloud", key="cam_speak"):
                audio = speak(sentence)
                st.audio(audio)
        else:
            st.warning("No Braille cells detected. Try a clearer image.")
