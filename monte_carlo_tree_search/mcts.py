import numpy as np

from config import Config


class MCTS:
    def __init__(self, network):
        self.network = network
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times state s was visited
        self.initial_policy = {}  # returned by neural net
        self.end_states = {}  # stores the outcome of finished games
        self.valid_moves = {}  # stores valid moves for a given state
    
    def get_raw_action_prob(self, game):
        policy, value = self.network.test(game)
        print(policy, value)
        return policy

    def get_action_prob(self, game, probabilistic=True):
        """perform numMCTSSims simulations of MCTS"""
        for _ in range(Config.numMCTSSims):
            self.search(game)

        state = game.string_representation()
        counts = [
            self.Nsa.get((state, action), 0) for action in range(game.get_action_size())
        ]

        if probabilistic:
            if sum(counts) != 0:
                return [x / sum(counts) for x in counts]
            # TODO: understand this case (no valid actions)

        probs = [0] * len(counts)
        probs[np.argmax(counts)] = 1
        return probs

    def search(self, game, player=1):
        state = game.string_representation()

        if state not in self.end_states:
            if game.is_over():
                self.end_states[state] = game.get_score()
            else:
                self.end_states[state] = None

        if self.end_states[state] is not None:  # reached terminal node
            return self.end_states[state]

        if state not in self.initial_policy:
            self.initial_policy[state], value = self.network.test(game)
            valids = game.get_valid_moves()
            # keep only valid moves
            self.initial_policy[state] = self.initial_policy[state] * valids
            # renormalize
            self.initial_policy[state] /= np.sum(self.initial_policy[state])

            # TODO: print policy below if verbose
            # print("policy", np.array2string(self.initial_policy[state], precision=2, suppress_small=True))

            self.valid_moves[state] = valids
            self.Ns[state] = 0
            return value

        # if the state has already been visited, then pick the action with the highest upper confidence bound
        actions = [
            action
            for action in range(game.get_action_size())
            if self.valid_moves[state][action]
        ]

        calculate_ucb = (
            lambda a: self.Qsa[(state, a)]
            + Config.cpuct
            * self.initial_policy[state][a]
            * np.sqrt(self.Ns[state])
            / (1 + self.Nsa[(state, a)])
            if (state, a) in self.Qsa
            else (
                Config.cpuct
                * self.initial_policy[state][a]
                * np.sqrt(self.Ns[state] + 1e-8)
            )
        )
        if player == 1:
            action = max(actions, key=calculate_ucb)
        else:
            action = min(actions, key=calculate_ucb)

        next_s, next_player = game.get_next_state(player, action)
        value = self.search(next_s, next_player)

        if (state, action) in self.Qsa:
            self.Qsa[(state, action)] = (
                self.Nsa[(state, action)] * self.Qsa[(state, action)] + value
            ) / (self.Nsa[(state, action)] + 1)
            self.Nsa[(state, action)] += 1

        else:
            self.Qsa[(state, action)] = value
            self.Nsa[(state, action)] = 1

        self.Ns[state] += 1

        return value
