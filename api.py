
"""
This module contains the KnisterGame class, which implements the rules and state
transitions of a Knister game.
"""

from __future__ import annotations
from typing import List

import numpy as np
import random


class KnisterGame:
    """
    Class implementing the full Knister game loop.

    This class manages the game state, including the grid, available actions,
    dice rolls, and scoring. It provides methods to roll the dice, select an
    action corresponding to a grid position, and track rewards incrementally
    as the game progresses, as well as the final total score.
    """

    SIZE: int = 5

    # Score constants
    FIVE_OF_A_KIND: int = 10
    FOUR_OF_A_KIND: int = 6
    FULL_HOUSE: int = 8
    THREE_OF_A_KIND: int = 3
    TWO_PAIRS: int = 3
    ONE_PAIR: int = 1
    STRAIGHT_WITH_SEVEN: int = 8
    STRAIGHT_NO_SEVEN: int = 12
    DIAGONAL_MULTIPLIER: int = 2

    def __init__(self) -> None:
        """
        Initialize a new Knister game instance.
        """
        self.grid: np.ndarray = np.zeros((self.SIZE, self.SIZE), dtype=int)
        self.available_positions: List[int] = list(range(self.SIZE * self.SIZE))
        self.current_roll: int | None = None
        self.finished: bool = False
        self.last_score: int = 0
        self.prev_score: int = 0

    def new_game(self) -> None:
        """
        Reset the game state and roll the first dice total.

        Returns:
            None
        """
        self.grid[:] = 0
        self.available_positions = list(range(self.SIZE * self.SIZE))
        self.current_roll = None
        self.finished = False
        self.last_score = 0
        self.prev_score = 0
        self.roll_dice()
        self.pool = []

    def roll_dice(self) -> None:
        """
        Roll two six sided dice and store their sum.

        The resulting value is an integer in [2, 12]. The distribution is not uniform:
        7 is the most probable outcome, while 2 and 12 are the least probable.

        Returns:
            None
        """
        self.current_roll = random.randint(1, 6) + random.randint(1, 6)

    def set_current_roll(self, value: int) -> None:
        """
        Manually set the current dice total.

        Args:
            value: Dice sum to set. Expected range is [2, 12].

        Returns:
            None
        """
        self.current_roll = value

    def get_current_roll(self) -> int | None:
        """
        Get the current dice total.

        Returns:
            The current dice sum, or None if not set.
        """
        return self.current_roll

    def get_grid(self) -> np.ndarray:
        """
        Get a copy of the game grid.

        Returns:
            A copy of the current grid.
        """
        return self.grid.copy()

    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Returns:
            A list of free cell indices.
        """
        return self.available_positions.copy()

    def choose_action(self, action: int) -> None:
        """
        Place the current dice total in the selected cell and update the game state.

        Args:
            action: Cell index where the value is placed.

        Returns:
            None

        Raises:
            GameFinished: If an action is attempted after the game has finished.
            InvalidAction: If the selected action is not available.
            NoDice: If no current dice roll is set.
        """
        if self.finished:
            raise GameFinished("Game has already finished")

        if action not in self.available_positions:
            raise InvalidAction(f"Invalid action: {action}")

        if self.current_roll is None:
            raise NoDice("Current roll is not set. Call roll_dice() or set_current_roll().")

        row, col = divmod(action, self.SIZE)
        self.grid[row, col] = self.current_roll
        self.available_positions.remove(action)

        score_now = self.calculate_score()
        self.last_score = score_now - self.prev_score
        self.prev_score = score_now

        if not self.available_positions:
            self.finished = True
        else:
            self.roll_dice()

    def has_finished(self) -> bool:
        """
        Check whether the game has finished.

        Returns:
            True if the game is finished, False otherwise.
        """
        return self.finished

    def get_last_reward(self) -> int:
        """
        Get the reward obtained by the last action.

        Returns:
            Last step reward.
        """
        return self.last_score

    def get_total_reward(self) -> int:
        """
        Get the total accumulated reward.

        Returns:
            Total score of the grid.
        """
        return self.calculate_score()

    def calculate_score(self) -> int:
        """
        Compute the total score of the current grid.

        Returns:
            Total score.
        """
        score = 0

        for i in range(self.SIZE):
            score += self.score_line(self.grid[i, :])
            score += self.score_line(self.grid[:, i])

        main_diag = np.diag(self.grid)
        anti_diag = np.diag(np.fliplr(self.grid))

        score += self.score_line(main_diag) * self.DIAGONAL_MULTIPLIER
        score += self.score_line(anti_diag) * self.DIAGONAL_MULTIPLIER

        return score

    def score_line(self, line: np.ndarray) -> int:
        """
        Compute the score of a single row, column, or diagonal.

        Args:
            line: Array representing the line to score.

        Returns:
            Score for the given line.
        """
        values = [v for v in line if v != 0]

        if len(values) < 2:
            return 0

        _, counts = np.unique(values, return_counts=True)
        counts_sorted = sorted(counts, reverse=True)

        if counts_sorted == [5]:
            return self.FIVE_OF_A_KIND
        if counts_sorted in ([4], [4, 1]):
            return self.FOUR_OF_A_KIND
        if counts_sorted == [3, 2]:
            return self.FULL_HOUSE
        if counts_sorted in ([2, 2], [2, 2, 1]):
            return self.TWO_PAIRS
        if counts_sorted[0] == 3:
            return self.THREE_OF_A_KIND
        if counts_sorted[0] == 2:
            return self.ONE_PAIR

        if len(values) == self.SIZE and len(set(values)) == self.SIZE:
            values_sorted = sorted(values)
            if all(values_sorted[i + 1] - values_sorted[i] == 1 for i in range(self.SIZE - 1)):
                return self.STRAIGHT_WITH_SEVEN if 7 in values_sorted else self.STRAIGHT_NO_SEVEN

        return 0


class KnisterException(Exception):
    """
    Base exception for all Knister game related errors.
    """


class GameFinished(KnisterException):
    """
    Raised when an action is attempted after the game has finished.
    """


class InvalidAction(KnisterException):
    """
    Raised when an invalid or unavailable action is selected.
    """


class NoDice(KnisterException):
    """
    Raised when an action is attempted without a current dice roll.
    """
