import torch
import torch.nn as nn
from models.base_model import BaseModel


class ResidualBlock(nn.Module):
    def __init__(self, n_dim: int = 128):
        super().__init__()

        self.conv1 = nn.Conv2d(n_dim, n_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_dim)
        self.conv2 = nn.Conv2d(n_dim, n_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_dim)

    def forward(self, x: torch.Tensor):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        out = nn.functional.relu(bn1)
        conv2 = self.conv2(out)
        bn2 = self.bn2(conv2)
        return nn.functional.relu(bn2 + x)


class PolicyValueModel(BaseModel):
    def __init__(
        self,
        board_size: int = 9,
        n_channels: int = 4,
        n_dim: int = 128,
        n_blocks: int = 5,
    ):
        super().__init__(board_size, n_channels)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(n_channels, n_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_dim),
            nn.ReLU(),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(n_dim) for _ in range(n_blocks)]
        )

        # Policy Head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(n_dim, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_linear = nn.Linear(2 * board_size * board_size, self.action_space)

        # Value Head
        self.value_conv = nn.Sequential(
            nn.Conv2d(n_dim, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_linear = nn.Sequential(
            nn.Linear(board_size * board_size, 2 * n_dim),
            nn.ReLU(),
            nn.Linear(2 * n_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv1 = self.conv_block1(x)
        res_blocks = self.residual_blocks(conv1)

        policy_conv = self.policy_conv(res_blocks)
        policy = policy_conv.flatten(start_dim=1)
        policy = self.policy_linear(policy)

        value_conv = self.value_conv(res_blocks)
        value = value_conv.flatten(start_dim=1)
        value = self.value_linear(value)

        return policy, value
