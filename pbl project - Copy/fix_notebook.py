import json

# Read the original Notebook structure
with open("Copy_of_SkinTone_Color_Advisor.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# The fixed code we prepared
fixed_code = """import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# Set dataset directory to match the unzipped folder in Colab
DATA_DIR = "skin_dataset/train"

# Load dataset from directory (split automatically)
print("Loading datasets...")
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
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) 
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
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
    epochs=10
)

# 💾 SAVE THE MODEL!
model.save("skin_tone_model.keras")
print("💾 Model successfully trained and saved as 'skin_tone_model.keras'!")
print("Run the next cell to download your model file from Colab!")
"""

# The Kaggle download commands cell
kaggle_setup = """# STEP 1: INITIALIZE DATASET 
!pip install -q kaggle
from google.colab import files

print("Please upload your kaggle.json file below:")
files.upload() # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d usamarana/skin-tone-classification-dataset
!unzip -q skin-tone-classification-dataset.zip -d skin_dataset
print("Dataset unzipped perfectly!")"""

download_cell = """# STEP 3: DOWNLOAD THE SAVED MODEL TO YOUR PC
from google.colab import files
files.download("skin_tone_model.keras")
"""

cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in kaggle_setup.split("\n")]
}

cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in fixed_code.split("\n")]
}

cell_3 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [line + "\n" for line in download_cell.split("\n")]
}

# Overwrite the notebook with the 3 clean, perfect cells
notebook["cells"] = [cell_1, cell_2, cell_3]

with open("Copy_of_SkinTone_Color_Advisor.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)
