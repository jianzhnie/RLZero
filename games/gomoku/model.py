import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """policy-value network module."""
    def __init__(self, board_width: int, board_height: int):
        super(PolicyValueNet, self).__init__()

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

    def forward(self, state):
        """
        Args:
            state(torch.Tensor): batch_size * channels * board_width * board_height
        """
        # common layers
        feat = F.relu(self.conv1(state))
        feat = F.relu(self.conv2(feat))
        feat = F.relu(self.conv3(feat))

        # action policy layers
        x_act = F.relu(self.act_conv1(feat))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        # batch_size x action_size
        policy_logits = self.act_fc1(x_act)
        policy_log_prob = F.log_softmax(policy_logits, dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(feat))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        # batch_size x 1
        x_val = self.val_fc2(x_val)
        value_out = torch.tanh(x_val)
        return policy_log_prob, value_out
