# Knister Game API

This repository contains a Python implementation of the Knister dice game, exposed through a simple and explicit API.

---

## The Knister Game

Knister is a turn based dice game played on a 5x5 grid.

Also known as *Würfel-Bingo*, it is such a notorious game that it only has a Wikipedia page in German. If you are curious (or brave), see [here](https://de.wikipedia.org/wiki/W%C3%BCrfel-Bingo).

### Dice mechanics

At each turn, two six sided dice are rolled and their sum is obtained.  
The possible values range from 2 to 12.

The probability distribution is not uniform:
- 7 is the most likely outcome
- 2 and 12 are the least likely outcomes

The rolled value must be placed into one empty cell of the grid.

### Game flow

1. The game starts with an empty 5x5 grid.
2. A dice roll is generated.
3. The player chooses one empty cell and places the roll value there.
4. The score is updated incrementally based on the new grid state.
5. Steps 2 to 4 repeat until all cells are filled.
6. The game ends when the grid is full.

---

## Scoring Rules

Scores are computed independently for:
- each row
- each column
- the two diagonals

Diagonal scores are multiplied by a constant factor.

Empty cells do not contribute to scoring.

### Combinations

For a given line (row, column, or diagonal):

| Combination                     | Score |
|---------------------------------|-------|
| Five of a kind                  | 10    |
| Four of a kind                  | 6     |
| Full house (3 + 2)              | 8     |
| Three of a kind                 | 3     |
| Two pairs                       | 3     |
| One pair                        | 1     |
| Straight of 5 values, no 7      | 12    |
| Straight of 5 values, with 7    | 8     |

A straight is valid only if:
- the line contains exactly five values
- all values are distinct
- values are consecutive

---

## API Overview

The main entry point is the `KnisterGame` class.

### Core responsibilities

The class:
- stores the game grid and available actions
- handles dice rolls or externally provided roll values
- applies player actions
- computes rewards incrementally
- computes the final total score

---

## Main Methods

### Game lifecycle

```
game = KnisterGame()
game.new_game()
```

Resets the game state and rolls the first dice total.

### Dice handling

```
game.roll_dice()  # Random roll
game.set_current_roll(value)  # Manual setting, debug use only
game.get_current_roll()
```

Dice rolls can be generated internally or set externally for deterministic evaluation.

### Actions

```
game.get_available_actions()
game.choose_action(action)
```

Actions correspond to grid cell indices from 0 to 24.

### State and rewards

```
game.get_grid()
game.get_last_reward()
game.get_total_reward()
game.has_finished()
```

Rewards are defined as the **incremental score change** caused by the last action.

Internally, the API tracks the total score before and after each placement.  
The value returned by `get_last_reward()` is computed as:

```
self.last_score = score_now - self.prev_score
```

For example, if placing a value upgrades a line from a one pair (`1` point) to a three of a kind (`3` points), the last reward returned is `2` (`3-1`).

A single placement may affect multiple rows, columns, or diagonals at once. In that case, all score increases and decreases are aggregated, and the returned reward reflects the net change in total score.

---

## Exceptions

The API uses explicit exceptions to signal invalid states:

- `InvalidAction`: the chosen action is not available
- `GameFinished`: an action was attempted after the game ended
- `NoDice`: an action was attempted without a current dice roll

All exceptions inherit from `KnisterException`.

---

## Manual Testing

The `play.py` module provides a simple command line interface to play the game manually.  
It is intended for debugging and verifying game logic, not as a polished user interface.

Run it with:

```
python play.py
```
