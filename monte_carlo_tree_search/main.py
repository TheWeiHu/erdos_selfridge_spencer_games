import argparse
import os
import numpy as np
import gym

from cnn import CNN
from ess_game import get_optimal_move
from train_routine import TrainRoutine
from mcts import MCTS
from config import Config


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def train():
    """main method"""
    network = CNN(Config.levels)
    TrainRoutine(network).learn()


def play(OPTIONS):
    """main method"""
    opponent = lambda x: np.random.choice(np.where(x.get_valid_moves() == 1)[0])
    num_games = 100

    if OPTIONS.optimal:
        opponent = get_optimal_move

    network = CNN(Config.levels)
    network.load_checkpoint("./temp/", "best.pth.tar")
    mcts = MCTS(network)

    trained_agent = lambda x: np.argmax(mcts.get_action_prob(x, False))

    a = gym.Gym(trained_agent, opponent, potential = 0.99) # consider setting potential = 0.95, 0.99, etc.
    print(a.play_games(num_games, verbose=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        dest="train",
        action="store_true",
        help="trains the agent (rather testing it).",
    )

    parser.add_argument(
        "--optimal",
        dest="optimal",
        action="store_true",
        help="plays against the optimal opponent.",
    )

    parser.set_defaults(optimal=False)
    parser.set_defaults(train=False)

    OPTIONS = parser.parse_args()

    if OPTIONS.train:
        train()
    else:
        play(OPTIONS)
