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

# Display first 25 images of dataset (for presentation purposes)
# plt.figure(figsize=(10,10))

# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])

# plt.show()

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
model.fit(train_images, train_labels, epochs=6)

