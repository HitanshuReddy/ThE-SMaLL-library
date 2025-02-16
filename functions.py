import tensorflow as tf

def ReLu(x):
    return tf.nn.relu(x)


def sigmoid(x):
    sigmoid = 1 / (1 + tf.exp(-x))
    return sigmoid

def tanh(x):
    return tf.tanh(x)

def leakyReLu(x):
    return tf.keras.layers.LeakyReLu(x)


def softmax(x):
    return tf.nn.softmax(x)

