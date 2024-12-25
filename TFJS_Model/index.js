import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest';

async function loadModel() {
    try {
        // Define the model architecture using TensorFlow.js syntax
        const model = tf.sequential();

        // Add layers to the model
        model.add(tf.layers.conv2d({
            filters: 16,
            kernelSize: [4, 4],
            inputShape: [28, 28, 1],
            activation: 'linear'
        }));

        model.add(tf.layers.maxPooling2d({
            poolSize: [4, 4],
            strides: [4, 4]
        }));

        model.add(tf.layers.conv2d({
            filters: 32,
            kernelSize: [2, 2],
            activation: 'linear'
        }));

        model.add(tf.layers.maxPooling2d({
            poolSize: [2, 2],
            strides: [2, 2]
        }));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: 128,
            activation: 'linear'
        }));

        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu'
        }));

        model.add(tf.layers.dense({
            units: 10,
            activation: 'softmax'
        }));

        console.log('Model loaded successfully!');
        model.summary();
    } catch (error) {
        console.error('Error loading the model:', error);
    }
}

loadModel();
