import numpy as np
from abc import ABC, abstractmethod
from env.board import Board
from mcts.mcts import MCTS
from mcts.evaluators import PolicyValueFn
from typing import Optional


class Player(ABC):
    def __init__(self, player_name: str = "John Doe"):
        self.player_name = player_name
        self.player_id = None

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def get_action(
        self,
        board: Board,
        tau: Optional[float] = None,
        add_noise: Optional[bool] = False,
    ) -> tuple[int, np.ndarray]:
        pass


class RandomPlayer(Player):
    def __init__(self, player_name: str = "Random", seed: Optional[int] = None):
        super().__init__(player_name)
        self.player_name = player_name
        self.rng = np.random.default_rng(seed)

    def get_action(
        self, board: Board, tau=None, add_noise=None
    ) -> tuple[int, np.ndarray]:
        legal_moves = board.get_legal_moves()
        return self.rng.choice(legal_moves), np.array(legal_moves)


class Agent(Player):
    def __init__(
        self,
        evaluator: PolicyValueFn,
        tau: float = 0.0,
        c_puct: float = 5.0,
        n_simulations: int = 500,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        player_name: str = "MCTS",
    ):
        super().__init__(player_name)
        self.player_id = None
        self.tau = tau
        self.mcts = MCTS(
            policy_value_fn=evaluator,
            c_puct=c_puct,
            n_simulations=n_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

    def reset(self):
        self.mcts.reset()

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    def get_action(
        self,
        board: Board,
        tau: Optional[float] = None,
        add_noise: Optional[bool] = True,
    ) -> tuple[int, np.ndarray]:
        if tau is None:
            tau = self.tau
        action, policy = self.mcts.search(board, tau, add_noise)

        return action, policy


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

    def get_action(self, board: Board, tau=None, add_noise=None):
        return self.input_retrieve(board), np.array([])
