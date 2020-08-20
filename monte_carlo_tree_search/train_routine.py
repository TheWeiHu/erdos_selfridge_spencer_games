import os
import pickle
import numpy as np

from config import Config
from collections import deque
from ess_game import ESSGame
from gym import Gym
from mcts import MCTS
from random import shuffle
from tqdm import tqdm


class TrainRoutine:
    def __init__(self, network):
        self.network = network  # current network
        self.previous_network = self.network.__class__(
            Config.levels
        )  # previous network
        self.mcts = MCTS(self.network)
        self.train_example_history = []

    def run_episode(self):
        train_examples = []
        game = ESSGame()
        episode_step = 0

        cur_player = 1

        while True:
            episode_step += 1
            probabilistic = episode_step < Config.probabilisticThreshold
            pi = self.mcts.get_action_prob(game, probabilistic)
            
            train_examples.append([game.board, pi])
            
            if max(pi) > pi[0] + 0.05:
                new_pi = [0] * len(pi)
                new_pi[0] = 1
                train_examples.append([game.swap_partitions().board, new_pi])

            action = np.random.choice(len(pi), p=pi)

            game, cur_player = game.get_next_state(cur_player, action)

            result = game.is_over()

            if result != 0:
                score = game.get_score()
                return [(x[0], x[1], score) for x in train_examples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        version = 0

        for i in range(Config.numIters):
            print(f"Iteration {i + 1}")
            iter_train_examples = deque([], maxlen=Config.maxlenOfQueue)

            print("Executing episodes...")
            for _ in tqdm(range(Config.numEps)):
                self.mcts = MCTS(self.network)  # reset search tree
                iter_train_examples.extend(self.run_episode())

            self.train_example_history.append(iter_train_examples)

            # backup history to a file
            self.save_train_examples(i)

            # shuffle examples before training
            train_examples = []
            for example in self.train_example_history:
                train_examples.extend(example)
            shuffle(train_examples)

            # training new network, keeping a copy of the old one
            self.network.save_checkpoint(Config.checkpoint, "temp.pth.tar")
            self.previous_network.load_checkpoint(Config.checkpoint, "temp.pth.tar")
            pmcts = MCTS(self.previous_network)

            self.network.train(train_examples)
            nmcts = MCTS(self.network)

            # play against previous version
            gym = Gym(
                lambda x: np.argmax(pmcts.get_action_prob(x, False)),
                lambda x: np.argmax(nmcts.get_action_prob(x, False)),
            )

            pwins, nwins, draws = gym.play_games(Config.gymCompare)

            print(f"new/prev wins: {nwins} / {pwins}, draws: {draws}")

            if pwins + nwins == 0 or nwins / (pwins + nwins) < Config.updateThreshold:
                self.network.load_checkpoint(Config.checkpoint, "temp.pth.tar")
            else:
                print(f"network version {version}")
                version += 1
                self.network.save_checkpoint(
                    Config.checkpoint, f"checkpoint_{i}.pth.tar"
                )
                self.network.save_checkpoint(Config.checkpoint, "best.pth.tar")
        
        print(f"num of versions: {version}")

    def save_train_examples(self, i):
        """save training examples"""
        folder = Config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f"checkpoint_{i}.pth.tar.examples")
        with open(filename, "wb+") as file:
            pickle.dump(self.train_example_history, file)
