import os
import numpy as np
import cv2
import time
from tqdm import tqdm
from collections import Counter
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ==========================
# 🔧 CONFIGURATION SECTION
# ==========================

DATASET_DIR = "synthetic_captchas"  # Directory containing CAPTCHA images
CAPTCHA_LENGTH = 6                 # Number of characters per CAPTCHA
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # Valid character set
IMG_WIDTH = 200                   # Resized image width
IMG_HEIGHT = 60                  # Resized image height
NUM_CLASSES = len(CHARACTERS)    # Total number of classification categories


# ==========================
# 📥 1. LOAD DATA FUNCTION
# ==========================

def load_data():
    """
    Loads CAPTCHA images from disk, normalizes them, and converts labels to indices.
    Also prints character distribution and skips malformed images.
    """
    X, y = [], []                             # Lists for image data and labels
    label_counter = Counter()                # Track frequency of characters
    files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"🔍 Loading {len(files)} images...")
    for fname in tqdm(files, desc="📥 Processing"):
        label = os.path.splitext(fname)[0].upper()  # Use filename as label
        if len(label) != CAPTCHA_LENGTH:
            tqdm.write(f"⚠️ Skipping {fname} — expected {CAPTCHA_LENGTH} chars, got {len(label)}")
            continue

        img_path = os.path.join(DATASET_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            tqdm.write(f"⚠️ Could not read image: {fname}")
            continue

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0  # Normalize to 0–1
        X.append(img)

        indices = [CHARACTERS.index(c) for c in label]  # Convert chars to class indices
        y.append(indices)
        label_counter.update(label)  # Count character occurrences

    # Display most frequent characters (helpful to verify data balance)
    print("\n📊 Most common characters in dataset:")
    for char, count in label_counter.most_common(10):
        print(f"  {char}: {count}")

    return np.array(X), np.array(y)


# ==========================
# 🧠 2. BUILD MODEL FUNCTION
# ==========================

def build_model():
    """
    Builds a CNN model using ResNet50 as base, followed by a shared dense layer,
    and six output softmax layers (one for each character).
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Freeze most layers to speed up training and avoid overfitting
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Input layer
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    # Feature extraction via pretrained ResNet50
    x = base_model(inputs, training=True)

    # Pooling + Dense layers for representation
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization

    # One output per character (6 in total)
    outputs = [
        layers.Dense(NUM_CLASSES, activation='softmax', name=f'char_{i+1}')(x)
        for i in range(CAPTCHA_LENGTH)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model using categorical cross-entropy loss
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"🧠 Model built. Total params: {model.count_params():,}")
    return model


# ==========================
# 🚀 3. TRAINING PIPELINE
# ==========================

if __name__ == "__main__":
    start_time = time.time()
    print("🔄 Starting CAPTCHA training pipeline...\n")

    # --- Load and preprocess the data ---
    X, y = load_data()
    y_list = [to_categorical(y[:, i], num_classes=NUM_CLASSES) for i in range(CAPTCHA_LENGTH)]

    print("\n📐 Data shapes:")
    print(f"  X: {X.shape}")
    for i, yi in enumerate(y_list):
        print(f"  y[{i+1}]: {yi.shape}")

    # --- Split data into training and validation sets ---
    split = train_test_split(
        X, *y_list, test_size=0.1, random_state=42
    )
    X_train, X_val = split[0], split[1]
    y_train_list = split[2::2]  # Even-indexed = train labels
    y_val_list = split[3::2]    # Odd-indexed = validation labels

    # --- Build and compile the model ---
    model = build_model()

    # --- Learning rate adjustment if no improvement ---
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    # --- Train the model ---
    print("\n🚀 Training model...\n")
    model.fit(
        X_train,
        y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=30,
        batch_size=64,
        callbacks=[lr_scheduler]
    )

    # --- Save the trained model ---
    model.save("captcha_resnet_model.h5")
    total_time = time.time() - start_time

    print(f"\n✅ Done. Model saved as captcha_resnet_model.h5")
    print(f"⏱️ Total training time: {total_time:.2f} seconds")
