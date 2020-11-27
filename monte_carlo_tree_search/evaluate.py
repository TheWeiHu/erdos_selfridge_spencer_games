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



def play(OPTIONS):
    """main method"""
    trained_agent = lambda x: np.argmax(mcts.get_action_prob(x, False))
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

    a = gym.Gym(random_player, trained_agent, potential=0.99)
    print(a.play_games(num_games, mode=0, verbose=True))


if __name__ == "__main__":
    for 