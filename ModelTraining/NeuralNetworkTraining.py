import keras
import numpy as np
import tensorflow as tf 

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
train_images = train_images.reshape((-1, 28, 28, 1))
train_labels = tf.keras.utils.to_categorical(train_labels, 10)

#def the model
MODEL = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
        keras.layers.MaxPool2D((3, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(784, input_dim=(28, 28, 1)),
        keras.layers.Dense(196, "relu"),
        keras.layers.Dense(49, "relu"),
        keras.layers.Dense(24, "relu"),
        keras.layers.Dense(10, "softmax")
        ])

#compile the model
MODEL.compile(
    optimizer=keras.optimizers.Adam(),
    loss='categorical_crossentropy',
    metrics=["accuracy"]
)
#Train the Model and save it

MODEL.fit(train_images, train_labels, epochs=30)
MODEL.save("MODEL.keras")


