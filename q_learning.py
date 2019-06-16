"""Q Learning Implementation

Implementation of the q learning algorithm using a q table. This approach is only
feasible for small boards.

Todo:
    * Consider adding type hints.

"""

import itertools
import math
import pickle
import random
import matplotlib.pyplot as plt
import game
import optimal
from lib import stars_and_bars as sab


def initialize_table(levels=4, pieces=15, filename="q_learning/qtable"):
    """Creates and a dictionary to represent a q table. The dictionary is then
    serialized using pickle and dumped into an output file.

    Args:
        levels: number of levels in the game.
        pieces: maximum number of pieces in the game.
        filename: the name of the file into which the serialized data is dumped.
    """
    outfile = open(filename, "wb")
    table = {}
    for num in range(1, pieces + 1):
        # Generates state space.
        for tup in sab.stars_and_bars(levels, num):
            # Calculates the size of the action space.
            action_size = 1
            for level in tup:
                action_size *= level + 1
            # Initializes q values for action space to be 1.
            table[tup] = [1] * action_size
    # Makes the value of the empty position 0.
    table[(0,) * levels] = [0]
    pickle.dump(table, outfile)
    outfile.close()


def initialize_weights(levels=4, near_sighted=True, filename="q_learning/weights"):
    """Creates and an array to represent the value function of a defender. The array is
    then serialized using pickle and dumped into an output file.

    Args:
        levels: number of levels in the game.
        near_sighted: generates the value function of a near-sighted defender if true,
        else generates the value function of a far-sighted defender.
        filename: the name of the file into which the serialized data is dumped.
    """
    outfile = open(filename, "wb")
    if near_sighted:
        weights = optimal.generate_near_sighted(levels)[1]
    else:
        weights = optimal.generate_far_sighted(levels)[1]
    pickle.dump(weights, outfile)
    outfile.close()


def load_table(filename="q_learning/qtable"):
    """Deserializes a dictionary representation of a q table.

    Args:
        filename: the name of the file from which the data is to be deserialized.
    Returns:
        the deserialized q table.
    """
    infile = open(filename, "rb")
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict


def load_weights(filename="q_learning/weights"):
    """Deserializes an array representation of a value function.

    Args:
        filename: the name of the file from which the data is to be deserialized.
    Returns:
        the deserialized value function.
    """
    infile = open(filename, "rb")
    weights = pickle.load(infile)
    infile.close()
    return weights


def save_table(table, filename="q_learning/qtable"):
    """A dictionary representation of a q table is serialized using pickle and dumped
    into an output file.

    Args:
        table: a dictionary representation of a q table.
        filename: the name of the file into which the serialized data is dumped.
    """
    outfile = open(filename, "wb")
    pickle.dump(table, outfile)
    outfile.close()


def sample_generator(iterator, items_wanted=1):
    """ Randomly samples outputs from an iterator (without consideration for order).

    Blatanly taken from:
    http://code.activestate.com/recipes/426332-picking-random-items-from-an-iterator/

    Args:
        iterator: iterator from which we sample a random outputs.
        item_wanted: number of items that are sampled.
    Returns:
        an array containing the sampled items.
    """
    selected_items = [None] * items_wanted
    for item_index, item in enumerate(iterator):
        for selected_item_index in range(items_wanted):
            if not random.randint(0, item_index):
                selected_items[selected_item_index] = item
    return selected_items


def generate_position(levels=4, pieces=15):
    """ Randomly generates an initial game position subject to the given constaints.
    (The time complexity for this approach is exponential; it could be worthwhile to
    implement an analytical (linear) solution).

    Args:
        levels: number of levels in the game.
        pieces: maximum number of pieces in the game.
    Returns:
        a random initial game position.
    """
    iterator = sab.stars_and_bars(levels, random.randint(1, pieces))
    return list(sample_generator(iterator)[0])


def get_action(state, qtable, explore=0.05):
    """ Selects an action from the q table based on the epsilon-greedy approach. In most
    cases we would select the action with the highest value. However, to allow
    exploration, we would occasionally select a random action.

    Args:
        state: the current state of the game.
        qtable: the q table from which we select actions.
        explore: the probability of choosing a random action.
    Returns:
        an action.
    """
    function = qtable[tuple(state)]
    if random.random() < (1 - explore):
        return function.index(max(function))
    return random.randint(0, len(function) - 1)


def output_action(state, action):
    """ Generates the partition corresponding to a given state and action.
    If an invalid action is provided, one of the sets will be the empty set.

    Args:
        state: the current state of the game.
        action: integer representation of an action.
    Returns:
        the corresponding partition.
    """
    # Do this part analytically, if necessary (reduce from exponential to linear).
    subsets = list(itertools.product(*(range(x + 1) for x in state)))
    if action >= len(subsets):
        action = 0
    partition = subsets[action]
    return list(partition), [i - j for i, j in zip(state, partition)]


def main():
    """  Implements the q learning algorithm using a q table. This approach is only
    feasible for small boards.
    """
    # Initializes q table and generates random defender value function.
    # initialize_table()
    # initialize_weights()

    # Training parameters
    number_of_games = 1000000
    discount = 0.95
    qtable = load_table()
    weights = load_weights()

    # Keeps track of gameplay quality.
    delta = []

    print(qtable)

    for iteration in range(number_of_games):
        # Generates game with random starting position.
        start_state = generate_position(4)
        env = game.Game(start_state, weights)
        score = 0
        while not env.is_finished():
            current_state = env.position
            # Selects an action from q table based on the epsilon-greedy algorithm.
            action = get_action(current_state, qtable)
            # Plays selected action.
            env.play(*output_action(current_state, action))
            new_score = env.score
            # Calculates new q value.
            new_q = (new_score - score) + discount * max(qtable[tuple(env.position)])
            # Updates table with new q value.
            row = qtable[tuple(current_state)]
            row[action] = new_q
            qtable[tuple(current_state)] = row
            score = new_score
        delta.append(env.score - math.floor(env.potential))

        # Makes a backup for every percentage of progress.
        if iteration % (number_of_games / 100) == 0:
            save_table(qtable)

    print(sum(delta) / len(delta))
    plt.plot(delta)
    plt.show()


if __name__ == "__main__":
    main()
