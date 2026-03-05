import numpy as np
from abc import ABC, abstractmethod
from env.board import Board
from typing import Optional


class Player(ABC):
    def __init__(self, player_name: str = "John Doe"):
        self.player_name = player_name
        self.player_id = None

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def get_action(self, board: Board) -> int:
        pass


class RandomPlayer(Player):
    def __init__(self, player_name: str = "Random", seed: Optional[int] = None):
        super().__init__(player_name)
        self.player_name = player_name
        self.rng = np.random.default_rng(seed)

    def get_action(self, board: Board) -> int:
        legal_moves = board.get_legal_moves()
        return self.rng.choice(legal_moves)


class HumanPlayer(Player):
    def input_retrieve(self, board: Board):
        while True:
            try:
                move = input(f"{self.player_name} >> ")
                row, col = move.split(",")
                row, col = int(row), int(col)
                idx = row * board.board_size + col
                if idx in board.availables:
                    return idx
                else:
                    print("Invalid move.")
            except (ValueError, IndexError):
                print("\tInvalid input. Enter row,col (e.g., 3,4).")

    def get_action(self, board: Board):
        return self.input_retrieve(board)


# TODO: AlphaZero-style Player
class AgentPlayer(Player):
    def get_action(self, board: Board):
        return -1
