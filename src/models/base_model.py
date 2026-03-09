import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from env.board import Board


class BaseModel(nn.Module, ABC):
    """
    Base Network Model for policy and value functions.
    `forward(x)` method outputs policy for all action space and state value in [-1, 1] range.
    """

    def __init__(self, board_size: int = 9, n_channels: int = 4):
        super().__init__()
        self.board_size = board_size
        self.n_channels = n_channels
        self.action_space = board_size * board_size
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            `policy`: `torch.Tensor` (batch_size, board_size * board_size)
            `value`: `torch.Tensor` (batch, 1)
        """
        pass

    def predict(self, board: Board) -> tuple[dict[int, float], float]:
        self.eval()

        with torch.no_grad():
            state = board.current_state()
            x = torch.tensor(state, dtype=torch.float32)
            x = x.unsqueeze(0).to(self.device)

            policy, value = self.forward(x)

            legal_moves = board.get_legal_moves()
            mask = torch.zeros(self.action_space, device=self.device)
            mask[legal_moves] = 1.0

            policy = torch.softmax(policy.squeeze(0), dim=0)
            policy = policy * mask
            policy = policy / policy.sum().clamp(min=1e-8)

            policy_np = policy.cpu().numpy()
            value_single = float(value.squeeze())

        action_prior_pairs = {move: float(policy_np[move]) for move in legal_moves}
        return action_prior_pairs, value_single
