import argparse
import os
import random
import numpy as np
import gym

from cnn import CNN
from ess_game import get_optimal_move
from train_routine import TrainRoutine
from mcts import MCTS
from config import Config


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# For importing near/far-sighted agents
"""
from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep + "suboptimal_play")

from optimal import far_sighted_algorithm, generate_far_sighted
_, weights = generate_far_sighted(Config.levels)
opponent = lambda x: far_sighted_algorithm(x.board, weights)
"""


def train():
    """main method"""
    network = CNN(Config.levels)
    TrainRoutine(network).learn()


def play(OPTIONS):
    """main method"""
    trained_agent = lambda x: np.argmax(mcts.get_raw_action_prob(x))
    random_player = lambda x: np.random.choice(np.where(x.get_valid_moves() == 1)[0])
    num_games = 100

    if OPTIONS.optimal:
        opponent = get_optimal_move
    elif OPTIONS.suboptimal:
        opponent = (
            lambda x: get_optimal_move(x)
            if random.random() > Config.suboptimality
            else random_player(x)
        )
    elif OPTIONS.suboptimal:
        opponent = trained_agent
    else:
        opponent = random_player

    network = CNN(Config.levels)
    network.load_checkpoint("./temp/", "best.pth.tar")
    mcts = MCTS(network)

    a = gym.Gym(trained_agent, opponent, potential=0.99)
    print(a.play_games(num_games, mode=1, verbose=True))


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

    parser.add_argument(
        "--suboptimal",
        dest="suboptimal",
        action="store_true",
        help="plays against the optimal opponent.",
    )

    parser.add_argument(
        "--selfplay",
        dest="selfplay",
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
