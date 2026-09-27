# webcam_skin_detect.py
import time
import cv2
import numpy as np
import tensorflow as tf
import math
from pathlib import Path
import joblib
from collections import deque
import pandas as pd

# ---------------- CONFIG ----------------
MODEL_FILE = "skin_tone_model.keras"
CLASS_NAMES = ["dark", "light", "medium"]
TARGET_SIZE = (160, 160)
WEBCAM_INDEX = 0
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------- SSD FACE DETECTOR ----------------
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ---------------- MULTI-FRAME ----------------
color_history = deque(maxlen=15)
tone_history = deque(maxlen=10)
undertone_history = deque(maxlen=10)

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
    bgr_pixel = np.uint8([[avg_rgb]])
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB)
    return lab_pixel[0][0]

def normalize_lighting(face):
    # CLAHE — only used for ML model input, NOT for LAB analysis
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# ✅ LAB based skin tone — corrected thresholds
def skin_tone_from_lab(L):
    if L < 60:
        return "dark"
    elif L < 75:
        return "medium"
    else:
        return "light"

def calculate_ita(L, b):
    if abs(b) < 1e-5:
        return 0
    return round(math.degrees(math.atan((float(L) - 50.0) / float(b))), 2)

def detect_undertone(a, b):
    if b > 15:
        return "Warm"
    elif a > 12 and b < 15:
        return "Cool"
    else:
        return "Neutral"

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
    voting_model = joblib.load("voting_model.pkl")
    print("Model loaded successfully.")

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

        (h, w) = frame.shape[:2]

        # -------- SSD DETECTION --------
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")

                width = x2 - x
                height = y2 - y

                # -------- FILTER (NO BOTTLES) --------
                if width < 100 or height < 100:
                    continue

                ratio = width / float(height)
                if ratio < 0.75 or ratio > 1.3:
                    continue

                # ✅ RAW face for LAB analysis (no brightness change)
                face_roi = frame[y:y2, x:x2]
                if face_roi.size == 0:
                    continue

                # -------- CENTER PATCH (raw, accurate colors) --------
                fh, fw = face_roi.shape[:2]
                cx1, cy1 = int(fw * 0.3), int(fh * 0.3)
                cx2, cy2 = int(fw * 0.7), int(fh * 0.7)
                skin_patch = face_roi[cy1:cy2, cx1:cx2]

                # ---- ML MODEL (normalized version only for ML) ----
                face_normalized = normalize_lighting(face_roi)
                batch = preprocess_frame(face_normalized)
                preds = model.predict(batch, verbose=0)

                # ---- COLOR ANALYSIS (raw patch — real skin colors) ----
                avg_rgb = np.mean(skin_patch.reshape(-1, 3), axis=0).astype(int)
                L_cv, a_cv, b_cv = rgb_to_lab(avg_rgb)
                
                L = (L_cv * 100.0) / 255.0
                a = a_cv - 128.0
                b = b_cv - 128.0

                # -------- MULTI-FRAME --------
                color_history.append((L, a, b))

                L_avg = np.mean([c[0] for c in color_history])
                a_avg = np.mean([c[1] for c in color_history])
                b_avg = np.mean([c[2] for c in color_history])

                ita = calculate_ita(L_avg, b_avg)
                undertone = detect_undertone(a_avg, b_avg)

                # ✅ HYBRID DECISION (LAB + Voting Classifier)
                features = pd.DataFrame(
                    [[L_avg, a_avg, b_avg, ita]],
                    columns=["L", "a", "b", "ITA"]
                )

                lab_tone = skin_tone_from_lab(L_avg)
                ml_tone = voting_model.predict(features)[0]

                # LAB trusted for clear cases, ML only for borderline
                if 60 <= L_avg <= 75:
                    skin_tone = ml_tone
                else:
                    skin_tone = lab_tone

                # -------- SMOOTHING --------
                tone_history.append(skin_tone)
                undertone_history.append(undertone)

                final_tone = max(set(tone_history), key=tone_history.count)
                final_undertone = max(set(undertone_history), key=undertone_history.count)

                # ---- COLOR RECOMMENDATION ----
                recommended_colors = COLOR_RECOMMENDATIONS.get(
                    (final_tone, final_undertone),
                    ["Black", "White"]
                )

                # ---- DRAW ----
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

                y_text = y - 10
                cv2.putText(frame, f"Skin Tone: {final_tone}",
                            (x, y_text), FONT, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Undertone: {final_undertone}",
                            (x, y_text - 25), FONT, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Colors: {', '.join(recommended_colors)}",
                            (x, y_text - 50), FONT, 0.5, (255, 255, 255), 1)

        # ---- FPS ----
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, frame.shape[0] - 10), FONT, 0.6, (200, 200, 200), 1)

        cv2.imshow("ToneStyle - Skin Tone & Color Recommendation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()