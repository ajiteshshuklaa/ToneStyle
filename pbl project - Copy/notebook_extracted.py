from google.colab import files
files.upload()


# --- CELL ---

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


# --- CELL ---

!kaggle datasets download -d usamarana/skin-tone-classification-dataset



# --- CELL ---


!unzip skin-tone-classification-dataset.zip -d skin_tone_data


# --- CELL ---

# STEP 5 — Verify files
import os
os.listdir("skin_tone_data")


# --- CELL ---

import pandas as pd
# Example (if there’s a CSV inside):
# df = pd.read_csv("skin_tone_data/filename.csv")


# --- CELL ---

import os

os.listdir("skin_tone_data")


# --- CELL ---

import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "skin_tone_data",
    image_size=(128, 128),
    batch_size=32
)

class_names = train_ds.class_names
print("Classes:", class_names)


# --- CELL ---

from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# --- CELL ---

history = model.fit(train_ds, epochs=10)


# --- CELL ---

import numpy as np

for images, labels in train_ds.take(1):
    predictions = model.predict(images)
    print("Predicted:", class_names[np.argmax(predictions[0])])


# --- CELL ---

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Step 1 — Collect all true and predicted labels
true_labels = []
pred_labels = []

for images, labels in train_ds:   # use test_ds if available
    predictions = model.predict(images)
    true_labels.extend(labels.numpy())
    pred_labels.extend(np.argmax(predictions, axis=1))

# Step 2 — Create confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Step 3 — Normalize the confusion matrix (optional)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Step 4 — Plot the confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Normalized)')
plt.show()

# Step 5 — Print detailed classification report
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))


# --- CELL ---

import tensorflow as tf

# Set dataset directory — change this to your actual folder
DATA_DIR = "/content/skin-tone-classification-dataset"  # example path

# Load dataset from directory (split automatically)
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


# --- CELL ---

!ls /content



# --- CELL ---

!pip install -q kaggle


# --- CELL ---

from google.colab import files
files.upload()  # Choose your kaggle.json file here


# --- CELL ---

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json


# --- CELL ---

!kaggle datasets download -d usamarana/skin-tone-classification-dataset
!unzip -q skin-tone-classification-dataset.zip -d /content/skin_dataset


# --- CELL ---

!ls /content/skin_dataset


# --- CELL ---

DATA_DIR = "/content/skin_dataset/train"


# --- CELL ---

import tensorflow as tf

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


# --- CELL ---

print(train_ds.class_names)


# --- CELL ---

from tensorflow.keras import layers, models

num_classes = 3  # based on the folders you just confirmed

# Build model
IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # freeze base

# Build final layers
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- CELL ---

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=10)
