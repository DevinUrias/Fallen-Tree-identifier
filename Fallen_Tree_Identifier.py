import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Constants
IMG_SIZE = 224
BATCH_SIZE = 29

# Paths
BASE_DIR = "Data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR  = os.path.join(BASE_DIR, "test")

# Load image dataset from CSV + folder
def load_dataset_from_csv(folder_path):
    df = pd.read_csv(os.path.join(folder_path, "labels.csv"))
    image_paths = [os.path.join(folder_path, fname) for fname in df['filename']]
    labels = df['fallen'].values

    def process_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_with_pad(image, IMG_SIZE, IMG_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    image_label_ds = path_ds.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    return image_label_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load datasets
train_ds = load_dataset_from_csv(TRAIN_DIR)
valid_ds = load_dataset_from_csv(VALID_DIR)
test_ds  = load_dataset_from_csv(TEST_DIR)

# Define model with data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

model = models.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

# Compile and train
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=valid_ds,
                    epochs=20)

# Evaluate on test set
loss, acc = model.evaluate(test_ds)
print(f"\n✅ Test Accuracy: {acc:.2%}\n")

# Generate predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    probs = model.predict(images).flatten()
    preds = (probs > 0.5).astype(int)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# Report results
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Not Fallen", "Fallen"], zero_division=0))

model.save("final_fallen_tree_model")
