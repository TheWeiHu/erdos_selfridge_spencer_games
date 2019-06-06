import math
import sys
import matplotlib.pyplot as plt
import optimal
import game


def plot(x, y, title="", x_label="", y_label=""):
    # Draw point based on above x, y axis values.
    plt.scatter(x, y, s=10)
    # Set chart title.
    plt.title(title)
    # Set x, y label text.
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def main():
    board_size = 2
    max_potential = 100

    potentials = []
    scores = []

    min_ratio = sys.maxsize
    max_ratio = 0
    min_position, max_position = None, None

    # Selects a random value function.
    weights = optimal.generate_far_sighted(board_size)[1]
    # Simulates games.
    for _ in range(1000):
        position = optimal.generate_far_sighted(board_size)[0]
        env = game.Game(position, weights)
        while env.potential < 1 or sum(env.position) == 1:
            position = optimal.generate_far_sighted(board_size)[0]
            env = game.Game(position, weights)
        while not env.is_finished():
            env.play(*optimal.far_sighted_algorithm(env.position, env.weights))
        if env.potential <= max_potential:
            potentials.append(env.potential)
            scores.append(env.score)
            ratio = env.score / env.potential
            if ratio < min_ratio:
                min_ratio = ratio
                min_position = [env.potential, env.score, position]
            if ratio > max_ratio:
                max_ratio = ratio
                max_position = [env.potential, env.score, position]
        if env.score < math.floor(env.potential):
            print("YELP!")
            exit()

    print(weights)
    print(min_position)
    print(max_position)

    # Plots Results.
    plot(potentials, scores)


if __name__ == "__main__":
    main()
