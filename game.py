"""Erdos Selfridge Spencer Game Environment

Erdos Selfridge Spencer Game environemnt for the attacking player.

Note: the board is zero-indexed: if not destroyed, a piece on the 0-th level will get
tenure this turn.

        TENURED
        ----------------------
        LEVEL 0
        -----------
        LEVEL 1
        -----------
        LEVEL 2
        -----------
        LEVEL 3
        -----------
            ...

Todo:
    * Consider adding type hints.

"""

import random
import optimal


class Game:
    """ Erdos Selfridge Spencer Game environemnt for the attacking player.

    Note: the board is zero-indexed: if not destroyed, a piece on the 0-th level will
    get tenure this turn.
    """

    def __init__(self, position, weights=None):
        self.score = 0
        self.position = position
        self.potential = optimal.position_value(self.position)
        # If there are no weights provided, the correct (optimal) weights are used.
        if not weights:
            self.weights = [1 / 2 ** (i + 1) for i in range(len(position))]
        else:
            self.weights = weights

    def play(self, position_1, position_2):
        """ Updates the game position, based on the defender evaluation of the the two
        partitions. The partition the defender deems more valuable is detroyed. The
        pieces of the surviving partition are promoted, and made into the game position.

        Args:
            position_1: the first of the two subsets.
            position_2: the second of the two subsets.
        """
        # Ensures that two subsets combined makes up the current position.
        assert [x + y for x, y in zip(position_1, position_2)] == self.position
        # Calculates the defender's evaluation of the two positions.
        value_1 = sum(i * j for (i, j) in zip(position_1, self.weights))
        value_2 = sum(i * j for (i, j) in zip(position_2, self.weights))
        # Keeps the partition, which according to the defender, has less value.
        if value_1 > value_2:
            self.position = position_2
        elif value_1 < value_2:
            self.position = position_1
        # Value of the two partition are equal --> randomly destroys one of the sets.
        elif random.randint(0, 1) == 1:
            self.position = position_1
        else:
            self.position = position_2
        self.promote()

    def promote(self):
        """ Promotes every piece on the board. If a pieces reaches tenured, it is
        removed, and the score is incremented by one.
        """
        self.score += self.position[0]
        # Selects the active, non-tenured pieces.
        new_position = list.copy(self.position[1:])
        # Promotes all active pieces.
        new_position.append(0)
        # Adds back the tenured pieces.
        self.position = new_position

    def is_finished(self):
        """Determines whether the game has finished.

        Returns:
            whether there are pieces remaining on the board.
        """
        return sum(self.position) == 0

    def to_string(self):
        """ Provides the current position, and a heuristic on how well the attacker has
        performed by evaluating how much "potential" score has been converted into
        real score.

        Returns:
            string representing of the position, the change in potential since the
            start of the game, and the score the attacker has obtained.
        """
        return (
            "Obtained Score: "
            + str(self.score)
            + "\nChange in Potential: "
            + str(optimal.position_value(self.position) - self.potential)
            + "\nGame Position: "
            + str(self.position)
            + "\n"
        )
