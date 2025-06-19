# 🐾 Animal Emotion Classifier

This project implements an AI-powered image classification system using **MobileNetV2** to detect and classify **animal emotions** from images. It includes preprocessing, data augmentation, training, validation, and real-time prediction using a file-based input system and Tkinter popups.

---

## 📌 Features

* 🧠 Built on **MobileNetV2** with transfer learning
* 🖼️ Processes and classifies images into 9 emotion classes
* 🔄 Includes **data augmentation** (rotation, zoom, flip, etc.)
* 🧪 Tracks validation accuracy using a built-in split
* 📊 Evaluation on separate test dataset
* 🖥️ Real-time batch prediction with Tkinter pop-up results

---

## 🗂️ Dataset Structure

Organize your data into:

```
datasets/
├── emotion_trainingdataset/
│   ├── class1/
│   ├── class2/
│   └── ...
├── emotion_testingdatasets/
│   ├── class1/
│   ├── class2/
│   └── ...
└── prediction_datasets/
    └── *.jpg / *.png / *.jpeg
```

Each class folder should contain labeled images corresponding to a specific animal emotion.

---

## 🚀 Getting Started

1. Clone the repository:

```bash
git clone https://github.com/yourusername/animal_emotion_classifier.git
cd animal_emotion_classifier
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the classifier:

```bash
python animal_emotion_classifier.py
```

---

## 🧠 Model Architecture

* **Base Model**: MobileNetV2 (frozen)
* **Classifier Head**:

  * GlobalAveragePooling2D
  * Dense(128, relu)
  * Dropout(0.5)
  * Dense(9, softmax)

Trained for 10 epochs using Adam optimizer and categorical crossentropy loss.

---

## 📈 Performance

* Trained on: 362 images
* Final Training Accuracy: \~95%
* Test Accuracy: \~91%

---

## 📦 Dependencies

* TensorFlow
* NumPy
* OpenCV (for display)
* Tkinter (for GUI popup)

---

## 🙋 Author

**Shivam Sharma**
GitHub: [@shivam15112003](https://github.com/shivam15112003)

---

Feel free to fork this repo, contribute, or adapt it for multi-species or behavior analysis projects!
