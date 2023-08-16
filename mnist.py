import tensorflow as tf
import numpy as np
from layers import Dense, Tanh, Sigmoid, Softmax


mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[1000:1500]
y_test = y_test[1000:1500]

network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

'''
Obtain the error the subtracting the output values of the network from the actual values in the labels and square this value.
Error values are summed up, constituting mean squared error of every iteration
'''
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def predict(network, input):
    output = input

    # Pass output from one layer into the next through the entire network
    for layer in network:
        output = layer.forward(output)

    return output

def train(network, loss, loss_derived, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for each in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss(y, output)
            gradient = loss_derived(y, output)

            # Perform backward propagation for current iteration
            for layer in reversed(network):
                gradient = layer.backward(gradient, learning_rate)

        error /= len(x_train)
        if verbose:
            print(f'{each + 1}/{epochs}, error = {error}')





