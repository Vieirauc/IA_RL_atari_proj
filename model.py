import numpy as np
import tensorflow as tf
import keras

from keras import layers


def build_model(height, width, channels, actions):
    model = keras.Sequential(
        [
            keras.Input(shape=(height, width, channels)),
            layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), activation="relu"),
            layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation="relu"),
            layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(actions, activation="linear"),
        ]
    )


    '''
    model = Sequential()
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3,height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    '''

    return model