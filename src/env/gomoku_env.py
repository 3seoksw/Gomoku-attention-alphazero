# WARNING: DEPRECATED
import gymnasium as gym
import numpy as np
from env.board import Board
from env.gomoku import Gomoku


class GomokuEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        board_size: int = 9,
        n_in_a_row: int = 5,
        start_player: int = 1,
        render_mode: str = "ansi",
    ):
        super().__init__()

        self.board_size = board_size
        self.n_in_a_row = n_in_a_row
        self.start_player = start_player
        self.render_mode = render_mode

        self.board = Board(board_size, n_in_a_row, start_player)
        self.game = Gomoku(self.board)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4, self.board_size, self.board_size),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(board_size * board_size)

    def _get_info(self, winner: int = -1) -> dict:
        return {
            "current_player": self.board.current_player,
            "legal_moves": self.board.availables,
            "move_counts": self.board.move_counts,
            "winner": winner,
        }

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.board.init_board(self.start_player)
        return self.board.current_state().copy(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        is_end, winner = self.game.step(action)

        reward = self._compute_reward(is_end, winner)
        obs = self.board.current_state().copy()
        if is_end:
            info = self._get_info(winner)
        else:
            info = self._get_info(-1)

        return obs, reward, is_end, False, info

    def render(self) -> None:
        if self.render_mode in ["human", "ansi"]:
            print(self.board)

    def close(self) -> None:
        pass

    def _compute_reward(self, is_end: bool, winner: int) -> float:
        if not is_end:
            return 0.0
        if winner == -1:
            return 0.0

        if self.board.current_player == self.board.players[0]:
            player_in_turn = self.board.players[1]
        else:
            player_in_turn = self.board.players[0]

        if player_in_turn == winner:
            return 1.0
        else:
            return -1.0
