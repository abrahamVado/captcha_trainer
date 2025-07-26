import os
import numpy as np
import string
from PIL import Image
from tensorflow.keras.models import load_model

# CONFIG
MODEL_PATH = "captcha_resnet_model.h5"
IMAGE_PATH = "synthetic_captchas/4CRUUV.jpeg"  # Replace with your image
IMAGE_SIZE = (200, 60)
CHARS = string.ascii_uppercase + string.digits
CAPTCHA_LENGTH = 6

# Load model
print("📥 Loading model...")
model = load_model(MODEL_PATH)

# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)  # Shape: (1, 60, 200, 3)

# Decode model output to text
def decode_prediction(preds, label_chars):
    result = ""
    for i in range(CAPTCHA_LENGTH):
        prob_vector = preds[i][0]  # (batch_size, n_classes) -> take first sample
        predicted_index = np.argmax(prob_vector)
        result += label_chars[predicted_index]
    return result

# Run prediction
print("🔍 Preprocessing image...")
input_img = preprocess_image(IMAGE_PATH)
print("🧠 Predicting...")
predictions = model.predict(input_img)

# Decode result
predicted_text = decode_prediction(predictions, CHARS)
print("✅ Predicted CAPTCHA:", predicted_text)
