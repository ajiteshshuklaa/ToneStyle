import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("skin_tone_model.keras")
skin_tone_labels = ["Light", "Medium", "Dark"]  # MUST match training order

def predict_skin_tone(face_img):
    # Resize to match training input
    face_img = cv2.resize(face_img, (224, 224))

    # Convert to float32
    face_img = face_img.astype(np.float32)

    # IMPORTANT: same preprocessing as training (MobileNet)
    face_img = tf.keras.applications.mobilenet_v2.preprocess_input(face_img)

    # Add batch dimension
    face_img = np.expand_dims(face_img, axis=0)

    # Predict
    prediction = model.predict(face_img, verbose=0)

    return skin_tone_labels[np.argmax(prediction)]
