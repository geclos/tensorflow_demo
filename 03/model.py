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
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
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
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Set some stuff (not particularly important)
model.compile(
    optimizer='adam', # Other examples of optimization: mini-batch, RMSProp, exponentially weighted averages, momentum... this is endless
    loss='sparse_categorical_crossentropy', # let's not enter into this...
    metrics=['accuracy']
)

# Train the model! :calculator:
model.fit(train_images, train_labels, epochs=1)

# Test it against the test set :pray:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Let's make some predictions
predictions = model.predict(test_images)
print('Predictions', predictions[0])
print('First prediction (most likely):', class_names[test_labels[np.argmax(predictions[0])]])

# Plot the first 10 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
plot_utils.plot_predictions(10, predictions, test_labels, test_images)

# save my model for reuse
model.save('fashion_model.h5')
