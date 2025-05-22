# app.py

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("models/mnist_model.keras")

# Set page title
st.set_page_config(page_title="MNIST Digit Classifier")

# Page header
st.title("✍️ Handwritten Digit Recognition")
st.markdown("Upload an image of a digit (28x28 or larger), and the model will predict it.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale and resize
    image = Image.open(uploaded_file).convert("L")
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized)

    # Invert colors if necessary
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize and reshape
    img_array = img_array / 255.0
    img_input = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_input)
    predicted_digit = np.argmax(prediction)

    # Show result
    st.success(f"✅ Predicted Digit: {predicted_digit}")
