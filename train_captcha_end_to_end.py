import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import load_model

# Configuration
DATASET_DIR = "synthetic_captchas"
CAPTCHA_LENGTH = 6
CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
IMG_WIDTH = 200
IMG_HEIGHT = 60
SEED = 42

def char_to_index(char):
    return CHARACTERS.index(char)

def load_data():
    X, y = [], []
    for fname in os.listdir(DATASET_DIR):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        label = os.path.splitext(fname)[0].upper()
        if len(label) != CAPTCHA_LENGTH:
            print(f"⚠️ Skipping {fname}: expected {CAPTCHA_LENGTH} chars, got {len(label)}")
            continue
        if any(c not in CHARACTERS for c in label):
            print(f"⚠️ Skipping {fname}: contains invalid character(s)")
            continue

        img_path = os.path.join(DATASET_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read image {fname}")
            continue
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
        X.append(img)
        y.append([char_to_index(c) for c in label])

    return np.array(X), np.array(y)

def build_model():
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)

    outputs = [
        layers.Dense(len(CHARACTERS), activation='softmax', name=f'char_{i}')(x)
        for i in range(CAPTCHA_LENGTH)
    ]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("🔍 Loading data...")
    X, y = load_data()
    if len(X) == 0:
        print("❌ No valid data found. Please check the dataset.")
        return

    y_list = [y[:, i] for i in range(CAPTCHA_LENGTH)]

    split_data = train_test_split(X, *y_list, test_size=0.1, random_state=SEED)

    X_train = split_data[0]
    X_val   = split_data[1]

    # Restructure label splits
    y_train_list = []
    y_val_list = []
    for i in range(CAPTCHA_LENGTH):
        y_train_list.append(split_data[2 + i * 2])
        y_val_list.append(split_data[2 + i * 2 + 1])


    print("🧠 Building model...")
    model = build_model()

    print("🚀 Training model...")
    model.fit(
        X_train,
        y_train_list,
        validation_data=(X_val, y_val_list),
        epochs=15,
        batch_size=64
    )

    model.save("captcha_model.h5")
    print("✅ Model saved as captcha_model.h5")

if __name__ == "__main__":
    main()
