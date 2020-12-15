import os 
import random
import numpy as np
import pandas as pd
import gym
import re

from cnn import CNN
from ess_game import get_optimal_move
from train_routine import TrainRoutine
from mcts import MCTS
from config import Config


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def play(mcts, potential=0.99, agent=0, mode=0):
    trained_agent = lambda x: np.argmax(mcts.get_raw_action_prob(x))
    random_player = lambda x: np.random.choice(np.where(x.get_valid_moves() == 1)[0])
    optimal_player = get_optimal_move
    mostly_random_player = (
        lambda x: get_optimal_move(x) if random.random() > 0.75 else random_player(x)
    )
    mostly_optimal_player = (
        lambda x: get_optimal_move(x) if random.random() > 0.25 else random_player(x)
    )
    agent_map = {
        0: optimal_player,
        1: mostly_random_player,
        2: mostly_optimal_player,
        3: random_player,
    }
    num_games = 100

    a = gym.Gym(trained_agent, agent_map[agent], potential=potential)
    return a.play_games(num_games, mode=mode, verbose=False)


if __name__ == "__main__":
    f = []
    for (dirpath, dirnames, filenames) in os.walk("./temp"):
        f.extend(filenames)
    f = [i for i in f if re.match(r"best-(\d)+\.pth\.tar", i)]
    print(f)

    agent_map = { 
        0: "optimal_player",
        1: "mostly_random_player",
        2: "mostly_optimal_player",
        3: "random_player",
    }

    for potential in [0.99, 1.99, 1.1, 2.1, 9.99]:

        a_res = pd.DataFrame(columns=["version"] + [i for i in agent_map.values()])
        d_res = pd.DataFrame(columns=["version"] + [i for i in agent_map.values()])

        for version in f:
            network = CNN(Config.levels)
            network.load_checkpoint("./temp/", version)
            mcts = MCTS(network)

            a_row = {"version": version}
            d_row = {"version": version}
            for i in range(4):
                a_row[agent_map[i]] = play(mcts, potential, i, 1)
                d_row[agent_map[i]] = play(mcts, potential, i, 2)

            a_res = a_res.append(a_row, ignore_index=True)
            d_res = d_res.append(d_row, ignore_index=True)

        a_res.to_csv(
            "./benchmark/attacker_" + str(potential) + ".csv",
            encoding="utf-8",
            index=False,
        )

        d_res.to_csv(
            "./benchmark/defender_" + str(potential) + ".csv",
            encoding="utf-8",
            index=False,
        )

