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

# Test it against the test set :pray:
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Let's make some predictions
predictions = model.predict(test_images)

print('Predictions', predictions[0])
print('First prediction (most likely):', class_names[test_labels[np.argmax(predictions[0])]])

# Plot the first 10 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols

plt.figure(figsize=(2*2*num_cols, 2*num_rows))

for i in range(10):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_utils.plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_utils.plot_value_array(i, predictions, test_labels)

plt.show()
