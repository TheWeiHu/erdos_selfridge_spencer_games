import numpy as np

from config import Config
from math import ceil, floor, log
from random import gauss, random


def piece_value(level):
    return 1 / 2 ** level


class ESSGame:
    def __init__(self, board=None, potential=None, n=Config.levels):
        self.n = n
        if board is None:
            self.initialize_random_2()
        else:
            self.board = board
        # TODO: ideally we find out where the floats are coming from
        self.board = self.board.astype(int)
        if potential is not None:
            self.potential = potential

    def swap_partitions(self):
        new_board = [self.board[0]]
        left, right = self.board[1 : (self.n + 1)], self.board[(self.n + 1) :]
        new_board.extend(right)
        new_board.extend(left)
        return ESSGame(np.array(new_board), self.potential)

    # TODO: consider training only on policy
    def symmetries_generator(self, max_distance=None, factors=[2, 3]):
        # TODO: distance and factor symmetries
        pass

    def initialize_random(self):
        n = self.n
        board = [0]
        for i in range(n):
            pieces = max(0, int(gauss(2 / piece_value(i), 1)))
            board.append(pieces)
        board += [0] * n
        self.board = np.array(board)
        self.initialize_potential()

    def initialize_random_2(self, potential=None):
        n = self.n
        if potential is None:
            potential = gauss(0.95, 0.75)
        board = [0] * (n + 1)
        while potential >= piece_value(n):
            piece = min(ceil(log(random(), 0.5)), n)
            if piece_value(piece) <= potential:
                board[piece] += 1
                potential -= piece_value(piece)
        board += [0] * n
        self.board = np.array(board)
        self.initialize_potential()

    def initialize_potential(self):
        total = 0
        for level, elem in enumerate(self.board):
            total += elem * piece_value(level)
        self.potential = floor(total)

    def get_action_size(self):
        return self.n + 1

    def get_next_state(self, player, action):
        new_board = np.copy(self.board)

        if player == 1:
            if not action:
                # the attacker chose the "done!" action
                if random() > 0.5:
                    return ESSGame(new_board, self.potential), -player
                # swap the order of the partitions with a 50% chance
                left, right = new_board[1 : (self.n + 1)], new_board[(self.n + 1) :]
                result = np.concatenate([[new_board[0]], right, left])
                return ESSGame(result, self.potential), -player
           
            # the attacker did not choose "done!"
            new_board = np.copy(self.board)
            assert new_board[action] > 0
            new_board[action] -= 1
            new_board[action + self.n] += 1
            # keep the same player, until finished
            return ESSGame(new_board, self.potential), player

        # TODO: check both directions
        elif player == -1:
            left, right = new_board[1 : (self.n + 1)], new_board[(self.n + 1) :]
            assert len(left) == len(right)

            result = right
            if not action:
                result = left

            # add back previous points
            result[0] += new_board[0]
            result = np.concatenate([result, np.zeros(self.n + 1)])
            return ESSGame(result, self.potential), -player

        raise Exception(f"Invalid player {player}")

    def get_valid_moves(self):
        """return vector of valid moves"""
        valids = [0] * self.get_action_size()

        moves = [i for i in range(self.n + 1) if self.board[i] > 0 or i == 0]
        for row in moves:
            valids[row] = 1
        return np.array(valids)

    def is_over(self):
        board = self.board
        if any(board[row] != 0 for row in range(1, self.n + 1)):
            return 0
        if board[0] == self.potential:
            return 1e-4
        if board[0] > self.potential:
            return 1
        if board[0] < self.potential:
            return -1

    def get_score(self):
        return self.board[0]

    def string_representation(self):
        return self.board.tostring() + ("p" + str(self.potential)).encode()

    def display(self):
        return np.array2string(self.board, precision=2)


def get_optimal_move(state):
    board = state.board
    n = len(board) // 2
    left, right = board[1 : (n + 1)], board[(n + 1) :]
    lp = rp = 0
    for level in range(n):
        lp += piece_value(level) * left[level]
        rp += piece_value(level) * right[level]
    valids = np.nonzero(left)[0]
    if valids.size == 0 or rp >= lp:
        return 0
    # choose smallest piece
    return np.max(valids) + 1


if __name__ == "__main__":
    game = ESSGame()
    game.initialize_random_2(0.99)
    print(game.board)
