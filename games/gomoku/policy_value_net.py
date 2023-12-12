import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """policy-value network module."""

    def __init__(self, board_width: int, board_height: int) -> None:
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(4 * board_width * board_height,
                                 board_width * board_height)
        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # common layers
        x = self.relu1(self.conv1(obs))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # action policy layers
        x_act = self.relu4(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        logits = self.act_fc1(x_act)
        x_act = F.log_softmax(logits, dim=1)

        # state value layers
        x_val = self.relu5(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = self.relu6(self.val_fc1(x_val))
        x_val = self.val_fc2(x_val)
        x_val = torch.tanh(x_val)
        return x_act, x_val
