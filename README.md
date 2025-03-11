# 🐾 Animal Detection Model

## 📌 Overview
This project implements an **AI-powered animal detection model** using **computer vision and deep learning**. It can identify various animal species and their emotions in images or videos, making it useful for **wildlife monitoring, security, and conservation efforts**.

## 🚀 Features
- **Real-time animal detection** in images and videos.
- Uses **MobileNetV2-based CNN model** for object classification.
- Supports **emotion recognition** for different animal species.
- Optimized for **high accuracy and low latency**.
- Can be integrated with **IoT devices** for smart surveillance.

## 🔧 Technologies Used
- **Python**
- **TensorFlow / Keras** (for deep learning and training the model)
- **OpenCV** (for image preprocessing and visualization)
- **NumPy** (for handling image arrays and data processing)
- **Tkinter** (for displaying prediction results via pop-up messages)

## 📂 Installation & Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/shivam15112003/animal_emotion_detection.git
   cd Animal_Detection_Model
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model:
   ```sh
   jupyter notebook emotional_detectionmodel.ipynb
   ```
4. Run real-time detection using a webcam:
   ```sh
   Run detection cells in the Jupyter notebook
   ```
5. Predict from an image:
   ```sh
   Use the Jupyter notebook for image predictions
   ```

## 📈 Model Training Details
- Uses **MobileNetV2** as a pre-trained model with additional custom layers.
- Trained on **9 different classes** with augmentation techniques.
- Achieved **91% accuracy** on the training dataset.

## 📈 Future Enhancements
- Improve model accuracy with more dataset training.
- Deploy as a **web-based API** for easy integration.
- Optimize for **edge devices** like Raspberry Pi.
- Implement real-time alert notifications for detected animals.
