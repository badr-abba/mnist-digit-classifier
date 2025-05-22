# predict_custom_image.py

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2

# Load the trained model in .keras format
model = load_model('models/mnist_model.keras')

# Path to your custom image (28x28 or larger)
img_path = 'samples/my_digit_3.png'

# Load the image in grayscale
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Resize to 28x28 pixels
img_resized = cv2.resize(img, (28, 28))

# Invert colors if necessary (MNIST is white text on black background)
if np.mean(img_resized) > 127:
    img_resized = 255 - img_resized

# Normalize the image to [0, 1]
img_normalized = img_resized / 255.0

# Reshape to match the input shape (1, 28, 28, 1)
img_input = np.expand_dims(img_normalized, axis=(0, -1))

# Predict the digit
prediction = model.predict(img_input)
predicted_digit = np.argmax(prediction)

# Show the result
plt.imshow(img_resized, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.axis('off')
plt.savefig("prediction_result.png")
plt.show()
