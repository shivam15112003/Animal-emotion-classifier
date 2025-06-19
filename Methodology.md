# 📊 Methodology: Animal Emotion Classification using MobileNetV2

This document describes the end-to-end methodology followed to build a deep learning model that classifies animal images into 9 emotional states using transfer learning.

---

## 1️⃣ Dataset Preparation

* Images were organized in `datasets/emotion_trainingdataset/` and `datasets/emotion_testingdatasets/` directories, each containing 9 folders corresponding to different animal emotion classes.
* Each folder consisted of \~50–150 high-quality images with clear class labeling.
* A separate folder `prediction_datasets/` was used for unseen image predictions.

---

## 2️⃣ Preprocessing & Augmentation

* Used `ImageDataGenerator` to rescale pixel values (1./255) and apply augmentations:

  * Rotation: ±30°
  * Width/Height shift: ±20%
  * Shear and Zoom transforms
  * Horizontal flip and brightness adjustment
* Applied 20% validation split during training for real-time performance tracking.

---

## 3️⃣ Model Architecture

* **Base Model**: `MobileNetV2` pretrained on ImageNet (224x224 input, `include_top=False`)
* **Transfer Learning**:

  * First \~100 layers frozen
  * Top 20 layers fine-tuned to learn emotion-specific features
* **Custom Classifier Head**:

  * `GlobalAveragePooling2D`
  * `Dense(128, relu)` + `Dropout(0.5)`
  * `Dense(9, softmax)` for multi-class output

---

## 4️⃣ Training Strategy

* Optimizer: `Adam` with learning rate `0.0001`
* Loss: `categorical_crossentropy`
* Epochs: 30 with early stopping
* Callbacks used:

  * `EarlyStopping(patience=5)` to prevent overfitting
  * `ModelCheckpoint` to save best model

---

## 5️⃣ Evaluation & Prediction

* Evaluated on a separate test set using accuracy metric
* Final Testing Accuracy: **91%** 
* Used trained model to predict images from `prediction_datasets/`
* Results displayed via Tkinter message box per image

---

## ✅ Outcome

* Efficient transfer learning system for animal emotion recognition
* Achieved fast and accurate inference with MobileNetV2 backbone
* GUI-ready prediction workflow integrated
* Suitable for real-time or offline animal behavior analysis applications

---

*Note: Performance results above are reported under a controlled test environment and may vary based on dataset scale and quality.*
