import os
import numpy as np

from keras.backend import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *

# TODO: further tune hyperparameters


class CNN:
    def __init__(self, levels):
        # two partitions, and an index to keep score
        size = 2 * levels + 1
        self.model = model_fn(size)

    def train(self, samples):
        states, target_policy, target_value = list(zip(*samples))
        states = np.asarray(states)
        target_policy = np.asarray(target_policy)
        target_value = np.asarray(target_value)
        self.model.fit(x=states, y=[target_policy, target_value])

    def test(self, state):
        state = np.asarray(state.board)
        # Enables batch of size one, for testing.
        state = state[np.newaxis, :]
        pi, v = self.model.predict(state)
        return pi[0], v[0]

    def save_checkpoint(self, folder="temp", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.model.save_weights(filepath)

    def load_checkpoint(self, folder="temp", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("no model found!")
        self.model.load_weights(filepath)


def model_fn(dimension):
    # Adds an extra dimension for to use convolutional layers.
    inputs = Input(shape=(dimension,))
    current = Reshape((dimension, 1))(inputs)

    # Builds the inner convolution layers.
    for _ in range(3):
        current = Conv1D(
            filters=128,
            kernel_size=3,
            kernel_initializer="Orthogonal",
            padding="same",
            kernel_regularizer=l2(0.01),
        )(current)
        current = Activation("relu")(
            BatchNormalization(axis=2, epsilon=0.0001)(current)
        )

    # Handles the extracted three dimensional features.
    current = Flatten()(current)
    for _ in range(2):
        current = Dropout(0.3)(
            Activation("relu")(
                BatchNormalization(axis=1, epsilon=0.0001)(
                    Dense(512, kernel_regularizer=l2(0.01))(current)
                )
            )
        )

    # Extracts outputs.
    policy = Dense(dimension // 2 + 1, activation="softmax", name="policy")(current)
    value = Dense(1, activation="tanh", name="value")(current)

    # Sets up loss function, and uses gradiant descent with Adam optimizer.
    model = Model(inputs=inputs, outputs=[policy, value])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(0.01)
    )
    return model


if __name__ == "__main__":
    m = CNN()
    print(m.model.summary())
