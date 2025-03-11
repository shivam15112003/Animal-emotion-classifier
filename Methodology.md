# 📌 Methodology

## 1️⃣ Data Collection & Preprocessing
- Collected animal images from various datasets.
- Resized and normalized images to fit the model’s input size.
- Applied data augmentation (rotation, zoom, flipping) to enhance model generalization.

## 2️⃣ Model Architecture & Training
- Used **MobileNetV2** as a pre-trained CNN model.
- Added custom layers for feature extraction and classification.
- Compiled the model using **Adam optimizer** and **categorical cross-entropy loss**.
- Trained on labeled datasets with **image augmentation**.

## 3️⃣ Emotion Detection & Classification
- Implemented **softmax activation** for multi-class classification.
- Predicted emotions based on extracted animal features.
- Evaluated accuracy using **training and validation sets**.

## 4️⃣ Deployment & Real-Time Detection
- Integrated OpenCV for **real-time video detection**.
- Developed a user-friendly Jupyter Notebook interface.
- Used Tkinter for pop-up alerts displaying detected emotions.

This methodology ensures **efficient, accurate, and real-time animal emotion detection** for applications in **wildlife monitoring, security, and automation**.
