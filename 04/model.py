from __future__ import absolute_import, division, print_function, unicode_literals

# plot utilities
import plot_utils

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# load dataset
fashion_mnist = keras.datasets.fashion_mnist
(_, _), (test_images, test_labels) = fashion_mnist.load_data()
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# preprocess data
# Scale to num between 0 and 1
test_images = test_images / 255.0

# Define the model
model = keras.models.load_model('fashion_model.h5')

# Review summary
model.summary()

# Test model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Let's make some predictions
predictions = model.predict(test_images)

print('Predictions', predictions[0])
print('First prediction (most likely):', class_names[test_labels[np.argmax(predictions[0])]])

# Plot the first 10 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
plot_utils.plot_predictions(10, predictions, test_labels, test_images)
