import torch
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 10000, batch_size: int = 64):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, self.batch_size)
        state, policy, value = zip(*batch)
        return (
            torch.stack(state),
            torch.stack(policy),
            torch.tensor(value, dtype=torch.float32),
        )
