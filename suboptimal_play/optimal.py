"""Optimal Algorithms for the Attacking Player

Algorithms generating optimal partitions for playing against suboptimal defenders.

Todo:
    * Consider adding type hints.

"""

import random
import itertools


def far_sighted_algorithm(position, weights):
    """Calculates the optimal partition given a position, and the value function of a
    far-sighted defender.

    Time Complexity:
        O(n), where n = sum(position).
    Args:
        position: vector representation of a game position, where position[i] represents
        the number of pieces on the i-th level of the board.
        weights: a value function, where weights[i] represents the value assigned to a
        piece on the i-th level of the board.
    Returns:
        the optimal partition.
    """
    # Ensures input is valid, and that the defender is indeed far-sighted.
    assert len(position) == len(weights)
    for i in range(len(weights) - 1):
        assert weights[i] < 2 * weights[i + 1]
    for level in position:
        assert level >= 0

    # Takes the dot product to calculate half of the total biased value.
    # The far-sighted defender will destroy any partition that has a biased value
    # greater than gamma.
    gamma = sum(i * j for (i, j) in zip(position, weights)) / 2

    optimal_set = [0] * len(position)
    cumulative_value = [(0, None)]
    current_index = sum(position)

    for i, level in reversed(list(enumerate(position))):
        for _ in range(level):
            cumulative_value.append((cumulative_value[-1][0] + weights[i], i))

    while gamma > 0 and current_index > 0:
        current_level = cumulative_value[current_index][1]
        next_cumulative_value = cumulative_value[current_index - 1][0]
        # Recursively, only includes a large piece when necessary: see proof.
        if next_cumulative_value - gamma < -1e-10: # Eliminates rounding errors.
            optimal_set[current_level] += 1
            gamma -= weights[current_level]
        current_index -= 1
    return optimal_set, [i - j for i, j in zip(position, optimal_set)]


def near_sighted_algorithm(position, weights):
    """Calculates the optimal partition given a position, and the value function of a
    near-sighted defender.

    Time Complexity:
        O(n), where n = sum(position).
    Args:
        position: vector representation of a game position, where position[i] represents
        the number of pieces on the i-th level of the board.
        weights: a value function, where weights[i] represents the value assigned to a
        piece on the i-th level of the board.
    Returns:
        the optimal partition.
    """
    # Ensures input is valid, and that the defender is indeed near-sighted.
    assert len(position) == len(weights)
    for i in range(len(weights) - 1):
        assert weights[i] > 2 * weights[i + 1]
    for level in position:
        assert level >= 0

    current_level = 0
    # Takes the dot product to calculate half of the total biased value.
    # The near-sighted defender will destroy any partition that has a biased value
    # greater than gamma.
    gamma = sum(i * j for (i, j) in zip(position, weights)) / 2
    optimal_value = position_value(position)
    optimal_set = position
    current_value = 0
    current_set = [0] * len(position)

    while current_level < len(position):
        if current_set[current_level] == position[current_level]:
            # Every piece in the current level has been included.
            # Moves on to the next level.
            current_level += 1
        elif weights[current_level] < gamma:
            # This piece is guarranteed to be in the next optimal partition: see proof.
            gamma -= weights[current_level]
            current_value += piece_value(current_level)
            current_set[current_level] += 1
        else:
            new_value = current_value + piece_value(current_level)
            new_set = current_set.copy()
            new_set[current_level] += 1
            current_level += 1
            # Checks to see whether this partition is more optimal partition.
            if new_value < optimal_value:
                optimal_value = new_value
                optimal_set = new_set
    return optimal_set, [i - j for i, j in zip(position, optimal_set)]


def brute_force_algorithm(position, weights):
    """Calculates an optimal partition given a position, and the value function of a
    defender.

    Time Complexity:
        O(2^n), where n = sum(position).
    Args:
        position: vector representation of a game position, where position[i] represents
        the number of pieces on the i-th level of the board.
        weights: a value function, where weights[i] represents the value assigned to a
        piece on the i-th level of the board.
    Returns:
        the optimal partition.
    """
    # Takes the dot product to calculate half of the total biased value.
    # The near-sighted defender will destroy any partition that has a biased value
    # greater than gamma.
    gamma = sum(i * j for (i, j) in zip(position, weights)) / 2
    optimal_value = position_value(position)
    optimal_set = position
    # Generates and tests every possible piece subset of the position.
    for partition in itertools.product(*(range(x + 1) for x in position)):
        if sum(i * j for (i, j) in zip(partition, weights)) > gamma:
            if position_value(partition) <= optimal_value:
                optimal_value = position_value(partition)
                optimal_set = partition
    return list(optimal_set), [i - j for i, j in zip(position, optimal_set)]


def piece_value(level):
    """Calculates the real value of a piece on a given level of the board.

    Args:
        level: a given level of the board.
    Returns:
        the real value of the piece on that level.
    """
    return 1 / 2 ** (level + 1)


def position_value(position):
    """Calculates the real value of a given position.

    Args:
        position: vector representation of a game position, where position[i] represents
        the number of pieces on the i-th level of the board.
    Returns:
        the real value of the position.
    """
    total = 0
    for level, elem in enumerate(position):
        total += elem * piece_value(level)
    return total


def generate_near_sighted(length=10, max_value=15, bias=1):
    """Creates a position and the value function of a near-sighted defender, based
    on given requirements.

    Args:
        length: number of levels in the game.
        max_value: maximum number of pieces per level.
        bias: the degree of near-sightedness.
    Returns:
        the real value of the position.
    """
    weights = [random.random()]
    position = [random.randint(0, max_value) for _ in range(length)]
    for _ in range(length - 1):
        weights.append(weights[-1] / (2 + random.random() * bias))
    return position, weights


def generate_far_sighted(length=10, max_value=15, bias=1):
    """Creates a position and the value function of a far-sighted defender, based
    on given requirements.

    Args:
        length: number of levels in the game.
        max_value: maximum number of pieces per level.
        bias: the degree of near-sightedness.
    Returns:
        the real value of the position.
    """
    weights = [random.random()]
    position = [random.randint(0, max_value) for _ in range(length)]
    for _ in range(length - 1):
        weights.append(weights[-1] / (2 - random.random() * bias))
    return position, weights


def main():
    """Tests implemented algorithms against the brute force algorithm to ensure correct
    implementation.
    """
    print("Near Sighted")
    for _ in range(10):
        near = generate_near_sighted(10, 5, 1)
        print(near_sighted_algorithm(*near))
        print(brute_force_algorithm(*near))

    print("Far Sighted")
    for _ in range(10):
        far = generate_far_sighted(10, 5, 1)
        print(far_sighted_algorithm(*far))
        print(brute_force_algorithm(*far))


if __name__ == "__main__":
    main()
