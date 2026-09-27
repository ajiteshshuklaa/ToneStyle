from PIL import Image, ImageOps
import cv2
import numpy as np
import tensorflow as tf
import joblib
import math
import os
import pandas as pd
from pathlib import Path

COLOR_MAP = {
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROTOTXT_PATH = str(BASE_DIR / "deploy.prototxt")
CAFFEMODEL_PATH = str(BASE_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

# Load face detector
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

# Load machine learning models
cnn_model = tf.keras.models.load_model("skin_tone_model.keras")


def get_face_roi(img):
    (h, w) = img.shape[:2]
    # Pass img directly instead of pre-resizing manually
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    best_confidence = 0
    best_box = None
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # Lowered threshold to 0.2 to capture rotated/distant faces in uploads
        if confidence > 0.2 and confidence > best_confidence:
            best_confidence = confidence
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            best_box = box.astype("int")
            
    if best_box is not None:
        (x, y, x2, y2) = best_box
        x = max(0, x)
        y = max(0, y)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return img[y:y2, x:x2]
    return img


def normalize_lighting(face):
    ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    y = clahe.apply(y)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def rgb_to_lab(avg_rgb):
    bgr_pixel = np.uint8([[avg_rgb]])
    lab_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2LAB) 
    return lab_pixel[0][0]


def analyze_image(image_path):
    try:
        # 0. Load with PIL to fix smartphone EXIF rotation metadata!
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        # Convert fixed image array back into OpenCV BGR format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        img = cv2.imread(image_path)
        
    if img is None:
        return "unknown", "unknown", ["Black", "White"], {"dark": 0, "medium": 0, "light": 0}

    # 1. Detect Face ROI to ignore body and background
    face_roi = get_face_roi(img)
    if face_roi.size == 0 or face_roi.shape == img.shape:
        # if face detection completely fails, gently crop center anyway to avoid background mostly
        fh, fw = img.shape[:2]
        face_roi = img[int(fh*0.2):int(fh*0.8), int(fw*0.2):int(fw*0.8)]

    # 2. Extract center patch for accurate color
    fh, fw = face_roi.shape[:2]
    cx1, cy1 = int(fw * 0.3), int(fh * 0.3)
    cx2, cy2 = int(fw * 0.7), int(fh * 0.7)
    skin_patch = face_roi[cy1:cy2, cx1:cx2]
    if skin_patch.size == 0:
        skin_patch = face_roi

    # 3. Predict ML model with standard RGB image (No lighting distortion)
    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (160, 160))
    batch = np.expand_dims(face_resized.astype(np.float32), axis=0)
    
    # Ensure MobileNet-style pixel preprocessing correctly passes into the network
    # Note: Keras natively handles preprocessing layers but using explicit is safer for MobileNet
    predictions = cnn_model.predict(batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    
    # Folders are 0: Black(dark), 1: Brown(medium), 2: White(light)
    cnn_tones = {0: "dark", 1: "medium", 2: "light"}
    tone = cnn_tones.get(predicted_idx, "medium")

    # 4. Color Analysis on raw non-normalized center patch for undertone
    avg_rgb = np.mean(skin_patch.reshape(-1, 3), axis=0).astype(int)
    L_cv, a_cv, b_cv = rgb_to_lab(avg_rgb)
    
    # Convert OpenCV 0-255 LAB scale to standard CIE LAB scale!
    L = (L_cv * 100.0) / 255.0
    a = a_cv - 128.0
    b = b_cv - 128.0
    
    # 5. Undertone
    if b > 15:
        undertone = "Warm"
    elif a > 12 and b < 15:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    colors = COLOR_MAP.get((tone, undertone), ["Black", "White"])
    
    scores = {
        "dark": round(float(predictions[0]) * 100, 1),
        "medium": round(float(predictions[1]) * 100, 1),
        "light": round(float(predictions[2]) * 100, 1)
    }
    
    return tone, undertone, colors, scores