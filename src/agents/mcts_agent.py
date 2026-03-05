from env.board import Board
from mcts.mcts import MCTS
from mcts.evaluators import RandomEvaluator
from typing import Optional


class MCTSAgent:
    def __init__(
        self,
        seed: Optional[int] = None,
        tau: float = 0.0,
        c_puct: float = 5.0,
        n_simulations: int = 500,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.player_id = None
        self.tau = tau
        self.mcts = MCTS(
            policy_value_fn=RandomEvaluator(seed),
            c_puct=c_puct,
            n_simulations=n_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

    def reset(self):
        self.mcts.reset()

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    def get_action(self, board: Board, tau: Optional[float] = None):
        if tau is None:
            tau = self.tau
        action, _ = self.mcts.search(board, tau=tau)

        return action
