# 🤖 CAPTCHA Solver with ResNet50

This project demonstrates how to train a custom **Convolutional Neural Network (CNN)** using **synthetic data** to solve simple alphanumeric CAPTCHA images — all **offline and self-hosted**.

## 🧩 Problem

A client needed to automate form inputs protected by image-based CAPTCHA challenges made of 6 uppercase letters and digits. Since the CAPTCHAs were not based on reCAPTCHA or hCaptcha, traditional solvers didn’t work — and sending data to external APIs was not an option.

---

## 💡 Solution

We built a complete training pipeline using Python and Keras:

- 🏗️ Synthetic CAPTCHA image generation (Pillow + random bubbles and noise)
- 🧠 CNN based on **ResNet50** backbone for multi-output classification
- 🧪 Offline training and validation using **TensorFlow/Keras**
- 🧾 CLI scripts for easy training, prediction, and dataset generation

---

## 📸 Example CAPTCHA

![Sample CAPTCHA](https://raw.githubusercontent.com/abrahamVado/captcha_trainer/main/preview.jpeg)

---

## 🚀 Features

- Generate unlimited synthetic training data
- Consistent font and layout for controlled experiments
- ResNet50 with 6 classification heads (1 per character)
- Trained with **categorical cross-entropy** and accuracy tracking per character
- Configurable character set: `A-Z`, `0–9`

---

## 🧪 How It Works

1. **Generate synthetic CAPTCHAs**

```bash
python synthetic_generation.py
```

2. **Train the model**

```bash
python train_captcha_ResNet50.py
```

3. **Predict with the trained model (optional)**

```bash
python predict.py --image path/to/captcha.jpg
```

---

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Or use the Windows batch setup:

```bash
setup.bat
```

---

## 📁 Project Structure

```
captcha_trainer/
├── synthetic_generation.py      # Generates synthetic images
├── train_captcha_ResNet50.py    # Trains the CNN model
├── predict.py                   # Loads a model and makes predictions
├── requirements.txt
├── setup.bat                    # Optional: installer + venv setup
├── synthetic_captchas/          # Generated dataset
└── captcha_resnet_model.h5      # Saved model (after training)
```

---

## 🧠 Concepts Used

- Convolutional Neural Networks (CNNs)
- Multi-output classification
- Data augmentation via synthetic generation
- Transfer learning with ResNet50
- Offline OCR via deep learning

---

## 📈 Model Performance (Example)

| Metric             | Accuracy |
|--------------------|----------|
| char_1_accuracy     | 16.5%    |
| char_6_accuracy     | 21.5%    |
| Overall trend       | Improving with data quality and epochs ✅ |

> Want higher accuracy? Train with more diverse fonts, colors, distortions, or noise!

---

## 📬 Contact & Contribute

This is a free and educational open-source project.

💻 **Try it. Fork it. Improve it.**

➡️ [https://github.com/abrahamVado//captcha_trainer](https://github.com/abrahamVado//captcha_trainer)

---

## 📜 License

MIT License — free to use, modify, and distribute.
