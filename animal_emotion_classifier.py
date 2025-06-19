# 🐾 Animal Emotion Classification using MobileNetV2 (Improved Version)

import tensorflow as tf
import numpy as np
import os
from tkinter import messagebox

# Set constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 9
PREDICTION_DIR = 'datasets/prediction_datasets'

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom classifier layers
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    'datasets/emotion_trainingdataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'datasets/emotion_trainingdataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Train the model
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

# Evaluate the model
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'datasets/emotion_testingdatasets',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

evaluation = model.evaluate(test_generator)
print('Testing Accuracy:', evaluation[1])

# Class label mapping
class_names = {v: k for k, v in train_generator.class_indices.items()}

# Predict new images
for filename in os.listdir(PREDICTION_DIR):
    if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
        continue
    image_path = os.path.join(PREDICTION_DIR, filename)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    predictions = model.predict(x)
    class_id = np.argmax(predictions)
    class_name = class_names[class_id]

    # Display prediction
    messagebox.showinfo("Prediction", f"Image: {filename}\nPredicted: {class_name}")
