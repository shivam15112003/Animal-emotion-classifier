import tensorflow as tf
import numpy as np
import os
import json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# ------------------ 1️⃣ DATA PREPROCESSING ------------------

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = 'datasets/emotion_trainingdataset'
TEST_DIR = 'datasets/emotion_testingdatasets'

# MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# ------------------ 2️⃣ BUILD MODEL ------------------

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Freeze first ~100 layers
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

# Custom classifier head
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=out)

# Compile with lower learning rate for fine-tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------ 3️⃣ TRAINING WITH CALLBACKS ------------------

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stop, model_checkpoint]
)

# ------------------ 4️⃣ EVALUATION ------------------

# Load best saved model
model = tf.keras.models.load_model('best_model.h5')

# Test data preprocessing
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Evaluate on test set
evaluation = model.evaluate(test_generator)
print(f'Testing Accuracy: {evaluation[1]*100:.2f}%')

# Save final model and class indices for GUI
model.save('animal_emotion_model_final.h5')

with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

print("✅ Model and class indices saved successfully!")


# ------------------ 5️⃣ GUI PREDICTION ------------------

def launch_gui():
    # Load model
    loaded_model = tf.keras.models.load_model('animal_emotion_model_final.h5')

    # Load class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)

    class_names = {v: k for k, v in class_indices.items()}

    # GUI window
    window = Tk()
    window.title("Animal Emotion Classification")
    window.geometry("600x600")
    window.config(bg="white")

    panel = Label(window)
    panel.pack(pady=20)

    pred_label = Label(window, text="", font=("Arial", 20), bg="white")
    pred_label.pack(pady=20)

    def select_image():
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        predict(file_path)

    def predict(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predictions = loaded_model.predict(x)
        class_id = np.argmax(predictions)
        class_name = class_names[class_id]
        confidence = predictions[0][class_id] * 100

        pred_label.config(text=f"Prediction: {class_name}\nConfidence: {confidence:.2f}%")

    btn = Button(window, text="Select Image", command=select_image, font=("Arial", 16), bg="lightblue")
    btn.pack(pady=20)

    window.mainloop()


# ------------------ 6️⃣ RUN GUI AFTER TRAINING ------------------
launch_gui()