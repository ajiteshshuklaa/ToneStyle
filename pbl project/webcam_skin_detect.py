# webcam_skin_detect.py
import time
import cv2
import numpy as np
import tensorflow as tf
import math
from pathlib import Path

# ---------------- CONFIG ----------------
MODEL_FILE = "skin_tone_model.keras"
CLASS_NAMES = ["dark", "light", "medium"]  # MUST match model training order
TARGET_SIZE = (160, 160)
WEBCAM_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ---------------------------------------

# ---------- LOAD MODEL ----------
def load_model_safe(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(str(p))

# ---------- PREPROCESS ----------
def preprocess_frame(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, TARGET_SIZE)
    batch = np.expand_dims(img_resized.astype(np.float32), axis=0)
    batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch)
    return batch

# ---------- COLOR SCIENCE ----------
def rgb_to_lab(avg_rgb):
    rgb_pixel = np.uint8([[avg_rgb]])
    lab_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2LAB)
    return lab_pixel[0][0]

def calculate_ita(L, b):
    if b == 0:
        return 0
    return round(math.degrees(math.atan((L - 50) / b)), 2)

def detect_undertone(a, b):
    if b > 15:
        return "Warm"
    elif a > 12 and b < 15:
        return "Cool"
    else:
        return "Neutral"

# ---------- FINAL SKIN TONE (ROBUST) ----------
def skin_tone_from_lab(L):
    """
    Final skin tone decision using LAB lightness.
    Much more stable than ITA under webcam lighting.
    """
    if L < 45:
        return "dark"
    elif 45 <= L < 60:
        return "medium"
    else:
        return "light"

# ---------- COLOR RECOMMENDATION ----------
COLOR_RECOMMENDATIONS = {
    ("light", "Warm"): ["Peach", "Cream", "Soft Coral", "Ivory"],
    ("light", "Cool"): ["Lavender", "Soft Blue", "Rose Pink"],
    ("light", "Neutral"): ["Blush", "Mint", "Light Grey"],

    ("medium", "Warm"): ["Mustard", "Olive", "Rust", "Coral"],
    ("medium", "Cool"): ["Emerald", "Teal", "Berry"],
    ("medium", "Neutral"): ["Teal", "Dusty Blue", "Soft Red"],

    ("dark", "Warm"): ["Gold", "Maroon", "Burnt Orange"],
    ("dark", "Cool"): ["Royal Blue", "Plum", "Emerald"],
    ("dark", "Neutral"): ["Charcoal", "White", "Crimson"]
}

# ---------- MAIN ----------
def main():
    print("Loading model...")
    model = load_model_safe(MODEL_FILE)
    print("Model loaded successfully.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    prev_time = time.time()
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            # ---- ML SKIN TONE (REFERENCE ONLY) ----
            batch = preprocess_frame(face_roi)
            preds = model.predict(batch, verbose=0)
            idx = int(np.argmax(preds))
            ml_skin_tone = CLASS_NAMES[idx]
            confidence = float(np.max(preds))

            # ---- SKIN COLOR ANALYSIS ----
            avg_rgb = np.mean(face_roi.reshape(-1, 3), axis=0).astype(int)
            L, a, b = rgb_to_lab(avg_rgb)
            ita = calculate_ita(L, b)
            undertone = detect_undertone(a, b)

            # ---- FINAL SKIN TONE (LAB-BASED) ----
            skin_tone = skin_tone_from_lab(L)

            # ---- COLOR RECOMMENDATION ----
            recommended_colors = COLOR_RECOMMENDATIONS.get(
                (skin_tone, undertone),
                ["Black", "White"]
            )

            # ---- DRAW OUTPUT ----
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            y_text = y - 10
            cv2.putText(frame, f"Skin Tone: {skin_tone}",
                        (x, y_text), FONT, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Undertone: {undertone}",
                        (x, y_text - 25), FONT, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Colors: {', '.join(recommended_colors)}",
                        (x, y_text - 50), FONT, 0.5, (255,255,255), 1)

        # ---- FPS ----
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, frame.shape[0]-10), FONT, 0.6, (200,200,200), 1)

        cv2.imshow("ToneStyle - Skin Tone & Color Recommendation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
