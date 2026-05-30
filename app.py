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

    # Find individual dots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dots = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if 20 < area < 500:
            cx = x + w // 2
            cy = y + h // 2
            dots.append((cx, cy))

    if not dots:
        return "", [], image_array

    # Auto calculate cell size from dot spacing
    xs = sorted(set(d[0] for d in dots))
    gaps = [xs[i+1] - xs[i] for i in range(len(xs)-1) if xs[i+1] - xs[i] > 3]
    CELL_W = int(np.median(gaps) * 2.5) if gaps else 30
    CELL_H = int(CELL_W * 1.5)

    used = set()
    cells = []
    dots_sorted = sorted(dots, key=lambda d: (d[0] // max(CELL_W,1), d[1] // max(CELL_H,1)))

    for i, (cx, cy) in enumerate(dots_sorted):
        if i in used:
            continue
        cell_dots = []
        for j, (ox, oy) in enumerate(dots_sorted):
            if j in used:
                continue
            if abs(ox - cx) < CELL_W and abs(oy - cy) < CELL_H * 1.5:
                cell_dots.append(j)
        for j in cell_dots:
            used.add(j)

        group_xs = [dots_sorted[j][0] for j in cell_dots]
        group_ys = [dots_sorted[j][1] for j in cell_dots]
x1 = max(0, min(group_xs) - 25)
y1 = max(0, min(group_ys) - 25)
x2 = min(image_array.shape[1], max(group_xs) + 25)
y2 = min(image_array.shape[0], max(group_ys) + 25)
        cells.append((x1, y1, x2, y2))

    if not cells:
        return "", [], image_array

    cells = sorted(cells, key=lambda c: c[0])

    result_letters = []
    result_confidences = []
    annotated = image_array.copy()

    for (x1, y1, x2, y2) in cells:
        cell = image_array[y1:y2, x1:x2]
        if cell.size == 0:
            continue
        letter, confidence = predict_cell(cell)
        result_letters.append(letter)
        result_confidences.append(confidence)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, letter.upper(), (x1, y1 - 5),
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
