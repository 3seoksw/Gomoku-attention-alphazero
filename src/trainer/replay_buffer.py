import torch
import random
from collections import deque


class ReplayBuffer:
    def __init__(
        self, capacity: int = 10000, batch_size: int = 64, device: str = "cuda"
    ):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size
        self.device = device

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, policy, value):
        self.buffer.append(
            (
                torch.from_numpy(state).float(),
                torch.from_numpy(policy).float(),
                torch.tensor(value, dtype=torch.float32),
            )
        )

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, self.batch_size)
        state, policy, value = zip(*batch)
        return (
            torch.stack(state).to(self.device),
            torch.stack(policy).to(self.device),
            torch.tensor(value, dtype=torch.float32).to(self.device),
        )
