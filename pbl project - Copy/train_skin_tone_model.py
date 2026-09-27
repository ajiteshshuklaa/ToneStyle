import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Set dataset directory
DATA_DIR = "archive (4)/train"

if not os.path.exists(DATA_DIR):
    print("❌ Dataset not found locally!")
    print("Please download the Kaggle dataset, unzip it, and place the images in 'skin_dataset/train'.")
    exit()

# Load dataset from directory (split automatically)
# IMPORTANT: image_size must strictly match the (160, 160) we use in predicting scripts
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160, 160),
    batch_size=16
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(160, 160),
    batch_size=16
)

class_names = train_ds.class_names
print("✅ Classes automatically detected:", class_names)
num_classes = len(class_names)

# Build model using MobileNetV2
IMG_SHAPE = (160, 160, 3)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze pre-trained weights

# Build final layers
inputs = tf.keras.Input(shape=IMG_SHAPE)
# Correct preprocessing applied natively to inputs
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x) # Help prevent overfitting
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
print("🚀 Training starting...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# SAVE THE MODEL 
# This was completely missing in the Colab notebook!
model.save("skin_tone_model.keras")
print("💾 Model successfully trained and saved as 'skin_tone_model.keras'!")
