import os
import numpy as np

from keras.backend import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.regularizers import *

# TODO: further tune hyperparameters


class CNN:
    def __init__(self, levels=5):
        # two partitions, and an index to keep score
        self.model = model_fn(levels)
        self.levels = levels

    def train(self, samples):
        states, target_policy, target_value = list(zip(*samples))
        boards = [transform_state(x, self.levels)[0] for x in states]
        scores = [transform_state(x, self.levels)[1] for x in states]
        boards = np.asarray(boards)
        scores = np.asarray(scores)
        target_policy = np.asarray(target_policy)
        target_value = np.asarray(target_value)
        self.model.fit(x=[boards, scores], y=[target_policy, target_value])

    def test(self, state):
        board, score = transform_state(state.board, self.levels)
        score = np.asarray(score)[np.newaxis, :]
        board = np.asarray(board)[np.newaxis, :]
        state = [board, score]
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


def transform_state(state, levels):
    return [state[1 : levels + 1], state[levels + 1 :]], [state[0]]


def model_fn(dimension):
    # Adds an extra dimension for to use convolutional layers.
    score = Input(shape=(1,))
    board = Input(shape=(2, dimension))
    current = Reshape((2, dimension, 1))(board)

    # Builds the inner convolution layers.
    current = Conv2D(
        filters=128,
        kernel_size=[2, 3],
        kernel_initializer="Orthogonal",
        padding="same",
        kernel_regularizer=l2(0.01),
    )(current)

    current = Activation("relu")(
        BatchNormalization(axis=2, epsilon=0.0001)(current)
    )

    # Handles the extracted three dimensional features.
    current = Flatten()(current)
    current = Dropout(0.3)(
        Activation("relu")(
            BatchNormalization(axis=1, epsilon=0.0001)(
                Dense(512, kernel_regularizer=l2(0.01))(current)
            )
        )
    )

    # Extracts outputs.
    merged = Concatenate()([current, score])
    policy = Dense(dimension + 1, activation="softmax", name="policy")(merged)
    value = Dense(1, activation="sigmoid", name="value")(merged)

    # Sets up loss function, and uses gradiant descent with Adam optimizer.
    model = Model(inputs=[board, score], outputs=[policy, value])
    model.compile(
        loss=["categorical_crossentropy", "mean_squared_error"], optimizer=Adam(0.01)
    )
    return model


if __name__ == "__main__":
    m = CNN(5)
    print(m.model.summary())
