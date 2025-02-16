from small import functions as f
from tensorflow import keras as ks
import numpy as np

## conv 2d network

def Conv2D(filter,size, x):
    """
    filter: Number of filters to use in the convolutional layer
    size: size of the convolutional filter
    x : target variable
    """
    return ks.layers.Conv2D(filter, size, activation='relu')(x)