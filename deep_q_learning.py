"""Deep Q Learning Implementation

Implementation of the q learning algorithm using a neural network. This approach is only
feasible for small boards.

Inspiration:
github.com/awjuliani/4d69edad4d0ed9a5884f3cdcf0ea0874#file-q-net-learning-clean-ipynb

Todo:
    * Consider adding type hints.

"""

import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import q_learning as q
import game
from lib import buckets

tf.reset_default_graph()

# Neural network parameters:
N_PIECES = 15
N_LEVELS = 4
N_HIDDEN_1 = 256
N_HIDDEN_2 = 256

# Initializes weights of neural network:
W1 = tf.Variable(tf.random_normal([N_LEVELS, N_HIDDEN_1]))
W2 = tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2]))
O = tf.Variable(
    tf.random_normal([N_HIDDEN_2, buckets.maximum_choices(N_PIECES, N_LEVELS)])
)

# Training parameters:
N_GAMES = 100000
L_RATE = 1e-7
DISCOUNT = 0.95
EPSILON = 0.95
# Load in the weight of a defender:
D_WEIGHTS = q.load_weights("deep_q_learning/weights")


def neural_net(input_state):
    """
    Neural network approximation of q table.
    """
    # Two fully connected layers:
    level_1 = tf.matmul(tf.expand_dims(input_state, 0), W1)
    level_2 = tf.matmul(level_1, W2)
    # One output neuron for each possible action:
    out = tf.matmul(level_2, O)
    return out


INPUT_STATE = tf.placeholder(shape=[N_LEVELS], dtype=tf.float32)


def main():
    """  Implements the q learning algorithm using a neural network. This approach is
    only feasible for small boards.
    """

    # Generates random defender value function.
    # initialize_weights("deep_q_learning/weights")

    q_values = neural_net(INPUT_STATE)
    predicted = tf.argmax(q_values, 1)
    next_q_values = tf.placeholder(
        shape=[1, buckets.maximum_choices(N_PIECES, N_LEVELS)], dtype=tf.float32
    )

    loss = tf.reduce_mean(tf.square(next_q_values - q_values))
    update = tf.train.GradientDescentOptimizer(learning_rate=L_RATE).minimize(loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    ratio = []

    # Starts training.
    with tf.Session() as sess:

        saver.restore(sess, "./deep_q_learning/model.ckpt")
        # init = tf.global_variables_initializer()
        # Runs the initializer.
        # sess.run(init)

        for iteration in range(N_GAMES):
            # Generates game with random starting position.
            env = game.Game(q.generate_position(1) + [0, 0, 0], D_WEIGHTS)
            score = 0
            while not env.is_finished():
                current_state = env.position
                # Selects an action from q table based on the epsilon-greedy algorithm.
                action, all_q_values = sess.run(
                    [predicted, q_values], feed_dict={INPUT_STATE: current_state}
                )
                # Plays selected action.
                env.play(*q.output_action(current_state, action[0]))
                # Gets the predicted q values of the next state.
                next_values = sess.run(q_values, feed_dict={INPUT_STATE: env.position})
                print(action)
                all_q_values[0, action[0]] = (env.score - score) + DISCOUNT * np.max(
                    next_values
                )
                sess.run(
                    [update, W1, W2, O],
                    feed_dict={INPUT_STATE: current_state, next_q_values: all_q_values},
                )
                score = env.score
            ratio.append(env.score - math.floor(env.potential))

            # Makes a backup for every percentage of progress.
            if iteration % (N_GAMES / 10) == 0:
                saver.save(sess, "./deep_q_learning/model.ckpt")
                print(sum(ratio)/len(ratio))
                plt.plot(ratio)
                plt.show()


if __name__ == "__main__":
    main()
