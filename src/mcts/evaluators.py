import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod
from env.board import Board
from models.base_model import BaseModel

"""
Evaluator class interface (signature)
Args:
    action_priors: dict[int, float] {action (move): prior}
    value: float
"""
PolicyValueFn = Callable[["Board"], tuple[dict[int, float], float]]


class BaseEvaluator(ABC):
    @abstractmethod
    def __call__(self, board: Board) -> tuple[dict[int, float], float]:
        """
        Evaluate a board position.
        Args:
            board: Board
        Returns:
            action_prior_pairs: dict[int, float]
            value: float
        """


class RandomEvaluator(BaseEvaluator):
    """
    Vanilla MCTS evaluator with random rollout
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _rollout(self, board: Board) -> float:
        """Random simulation"""
        b = board.clone()
        player = b.get_current_player()
        while True:
            is_end, winner = b.is_game_end()
            if is_end:
                if winner == -1:  # draw or ongoing
                    return 0.0
                if winner == player:  # win
                    return 1.0
                else:  # lose
                    return -1.0
            move = int(self.rng.choice(b.get_legal_moves()))
            b.play_move(move)

    def __call__(self, board: Board) -> tuple[dict[int, float], float]:
        legal_moves = board.get_legal_moves()
        prior = 1.0 / len(legal_moves)
        action_prior_pairs = {}
        for move in legal_moves:
            action_prior_pairs[move] = prior
        value = self._rollout(board)
        return action_prior_pairs, value


class ModelEvaluator(BaseEvaluator):
    def __init__(self, model: BaseModel, device: str = "cuda"):
        self.model = model
        self.model.device = device

    def __call__(self, board: Board) -> tuple[dict[int, float], float]:
        return self.model.predict(board)
