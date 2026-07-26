"""
Manual test script for the KnisterGame API.

This module provides a simple command line interface to play Knister by hand.
It is intended for debugging and manual verification of the game logic.
"""

import numpy as np

from api import KnisterGame


def print_grid(grid: np.ndarray):
    """Print the current grid state to the console."""
    print("\nGRIGLIA ATTUALE:")
    print("    1   2   3   4   5")
    print("  +---+---+---+---+---+")
    for r in range(5):
        row_vals = []
        for c in range(5):
            val = grid[r, c]
            row_vals.append(f"{val:2d}" if val != 0 else "  ")
        print(f"{r + 1} |" + " |".join(row_vals) + " |")
        print("  +---+---+---+---+---+")
    print()


def ask_action(game: KnisterGame):
    """Ask the user for a valid action and return it."""
    avail = game.get_available_actions()
    roll = game.get_current_roll()

    print(f"Valore corrente dei dadi: {roll}")
    print(f"Celle libere rimaste: {len(avail)}")

    while True:
        s = input(
            "Scegli una casella (indice 0-24 oppure 'r,c' con riga e colonna 1-5): "
        ).strip()

        if "," in s:
            # Row, column format
            try:
                r_str, c_str = s.split(",")
                r = int(r_str) - 1
                c = int(c_str) - 1

                if 0 <= r < 5 and 0 <= c < 5:
                    action = r * 5 + c
                    if action in avail:
                        return action
                    else:
                        print("Quella casella è già occupata, riprova.")
                else:
                    print("Riga/colonna fuori range (1-5), riprova.")
            except ValueError:
                print("Formato non valido, riprova (esempio valido: 2,3).")

        else:
            # Single index format
            try:
                idx = int(s)
                if idx in avail:
                    return idx
                else:
                    print("Indice non valido o casella occupata, riprova.")
            except ValueError:
                print("Input non valido, riprova.")


def main():
    """Run a manual Knister game session."""
    game = KnisterGame()
    game.new_game()

    print("Benvenuto a Knister!")
    print("Compila la griglia 5x5 piazzando il valore dei dadi nelle caselle libere.\n")

    while not game.has_finished():
        print_grid(game.get_grid())
        print(f"Reward dell'ultima mossa: {game.get_last_reward()}")
        print(f"Punteggio totale attuale: {game.get_total_reward()}\n")
        
        action = ask_action(game)
        game.choose_action(action)

    print(f"Reward dell'ultima mossa: {game.get_last_reward()}")
    print(f"Punteggio totale attuale: {game.get_total_reward()}\n")
    print("Partita terminata!")
    print_grid(game.get_grid())
    print(f"Punteggio finale: {game.get_total_reward()}")


if __name__ == "__main__":
    main()
