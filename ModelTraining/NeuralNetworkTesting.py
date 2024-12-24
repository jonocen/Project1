import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



test_images = test_images / 255.0
test_images = test_images.reshape((-1, 28, 28, 1))
test_labels = keras.utils.to_categorical(test_labels, num_classes=(10))


Model = keras.models.load_model("MODEL.keras")
predictions = Model.predict(test_images)

# Select the first 5 images from the test set for demonstration
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray') 
    plt.title(f"Predicted: {np.argmax(predictions[i])}, True: {np.argmax(test_labels[i])}")
    plt.show()