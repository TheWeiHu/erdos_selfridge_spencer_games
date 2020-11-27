from tqdm import tqdm

from ess_game import ESSGame

results = {1e-4: "Draw", 1: "Attacker Wins", -1: "Defender Wins"}


class Gym:
    def __init__(self, player1, player2, potential = None):
        self.player1 = player1
        self.player2 = player2
        self.potential = potential

    def play_game(self, verbose=False):
        """Execute an episode of a game:

        returns:
            1 if player1 wins, -1 if player2 wins, 0 if there is a draw
        """
        players = {1: self.player1, -1: self.player2}

        cur_player = 1
        game = ESSGame()
        if self.potential is not None:
            game.initialize_random_2(self.potential)

        while not game.is_over():

            if verbose:
                print(game.display())

            action = players[cur_player](game)
            game, cur_player = game.get_next_state(cur_player, action)

        if verbose:
            print(game.display())
            print("Outcome: " + results[game.is_over()])

        return game.is_over()

    def play_games(self, num, mode=0, verbose=False):
        """
        The mode variable determines which agent plays what role
        When:
            mode == 0, both players take turn playing both roles
            mode == 1, player 1 plays attacker, player 2 plays defender
            mode == 2, player 2 plays attacker, player 1 plays defender
        """
        num //= 2
        one_wins = 0
        two_wins = 0
        draws = 0

        if not mode or mode == 1:
            for _ in tqdm(range(num)):
                game_result = self.play_game(verbose=verbose)
                if game_result == 1:
                    one_wins += 1
                elif game_result == -1:
                    two_wins += 1
                else:
                    draws += 1

        self.player1, self.player2 = self.player2, self.player1


        if not mode or mode == 2:
            for _ in tqdm(range(num)):
                game_result = self.play_game(verbose=verbose)
                if game_result == -1:
                    one_wins += 1
                elif game_result == 1:
                    two_wins += 1
                else:
                    draws += 1

        return one_wins, two_wins, draws
