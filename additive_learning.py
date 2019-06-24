"""Additive Learning Implementation

We reduce the action space making it linear. Now, the Q learning approach can extend
to much larger boards.

Todo:
    * Consider adding type hints.

"""

import math
import random

# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import q_learning as q
import game

tf.reset_default_graph()

# Game parameters:
N_PIECES = 15
N_LEVELS = 4

# Neural network parameters:
N_HIDDEN_1 = 256
N_HIDDEN_2 = 256

# Initializes weights of neural network:
W1 = tf.Variable(tf.random_normal([N_LEVELS * 2, N_HIDDEN_1]))
W2 = tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2]))
O = tf.Variable(tf.random_normal([N_HIDDEN_2, N_LEVELS + 1]))

# Training parameters:
N_GAMES = 10000000
# Note that having a large learning rate leads to large updates, which fills the
# matrices with NAN.
L_RATE = 0.00000001
DISCOUNT = 0.95
EPSILON = 0.95
# Load in the weight of a defender:
D_WEIGHTS = q.load_weights("additive_learning/weights")


def neural_net(input_state):
    """ Attaches the different omponents of the Tensorflow graph forming the neural
    network approximation of the q table.

    Args:
        input_state: the current state of the game, stored in an input placeholder
        fed in from the game environment.
    Returns:
        the output of the network.
    """
    # Two fully connected layers:
    level_1 = tf.matmul(tf.expand_dims(input_state, 0), W1)
    level_2 = tf.matmul(level_1, W2)
    # One output neuron for each possible action:
    out = tf.matmul(level_2, O)
    return out


INPUT_STATE = tf.placeholder(shape=[N_LEVELS * 2], dtype=tf.float32)

# Keeps track of gameplay quality compared to the best expected value against an
# optimal defender.
DELTA = []


def main():
    """  Implements the q learning algorithm using a neural network. This approach is
    only feasible for small boards.
    """

    # Generates random defender value function.
    # initialize_weights("deep_q_learning/weights")

    q_values = neural_net(INPUT_STATE)
    predicted = tf.argmax(q_values, 1)
    next_q_values = tf.placeholder(shape=[1, N_LEVELS + 1], dtype=tf.float32)

    # Note reduce_sum() leads to large updates, which fills the matrices with NAN.
    loss = tf.reduce_mean(tf.square(next_q_values - q_values))
    update = tf.train.GradientDescentOptimizer(learning_rate=L_RATE).minimize(loss)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Starts training.
    with tf.Session() as sess:

        saver.restore(sess, "./additive_learning/model.ckpt")
        # Runs the initializer.
        # sess.run(tf.global_variables_initializer())

        for iteration in range(N_GAMES):
            # Generates game with random starting position.
            env = game.Game(q.generate_position(1) + [0, 0, 0], D_WEIGHTS)
            score = 0
            while not env.is_finished():
                # The current state consists of the two partitions appended, initialized
                # to having all the pieces in the first partition.
                current_state = env.position + [0] * N_LEVELS

                # Selects an action from q table based on the epsilon-greedy algorithm.
                while True:
                    action, all_q_values = sess.run(
                        [predicted, q_values], feed_dict={INPUT_STATE: current_state}
                    )
                    print(all_q_values)
                    # print(all_q_values)
                    if action[0] == N_LEVELS or current_state[action[0]] <= 0:
                        break
                    # Should epsilon greedy be done for the action as a whole instead?
                    if random.random() > EPSILON:
                        # Currently, most of the randomly chosen actions will be bad;
                        # perhaps, we should look into only selecting a valid action.
                        action = [random.randint(0, N_LEVELS - 1)]
                    current_state[action[0]] -= 1
                    current_state[N_LEVELS + action[0]] += 1
                    # Gets the predicted q values of the next state.
                    next_values = sess.run(
                        q_values, feed_dict={INPUT_STATE: current_state}
                    )
                    # print(action)
                    all_q_values[0, action[0]] = DISCOUNT * np.max(next_values)
                    # Applies gradient descent and update network.
                    sess.run(
                        [update, W1, W2, O],
                        feed_dict={
                            INPUT_STATE: current_state,
                            next_q_values: all_q_values,
                        },
                    )

                subset = current_state[N_LEVELS:]

                # If the move generated is invalid, refuse to partition.
                # (In otherwords, one subset will contain all the pieces).
                if any(x < 0 for x in [i - j for i, j in zip(env.position, subset)]):
                    subset = [0] * N_LEVELS

                # Plays selected action.
                env.play(list(subset), [i - j for i, j in zip(env.position, subset)])

                # Gets the predicted q values of the next state.
                next_values = sess.run(
                    q_values, feed_dict={INPUT_STATE: env.position + [0] * N_LEVELS}
                )
                all_q_values[0, action[0]] = (env.score - score) + DISCOUNT * np.max(
                    next_values
                )
                # Applies gradient descent and update network.
                sess.run(
                    [update, W1, W2, O],
                    feed_dict={INPUT_STATE: current_state, next_q_values: all_q_values},
                )
                score = env.score
            DELTA.append(env.score - math.floor(env.potential))

            # Makes a backup for every percentage of progress.
            if iteration % (N_GAMES / 10000) == 999:
                saver.save(sess, "./additive_learning/model.ckpt")
                # print(DELTA)
                print(sum(DELTA[-999:]) / 999)
                # plt.plot(DELTA)
                # plt.show()


if __name__ == "__main__":
    main()
