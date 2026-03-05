import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.MaxPool2d(3)

        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
