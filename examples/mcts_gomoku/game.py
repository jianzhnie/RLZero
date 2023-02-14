import numpy as np
import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, board_size):
        super().__init__()

        self.feat_net = nn.Sequential(nn.Conv2d(4, 32, 3, 1, 1), nn.ReLU(),
                                      nn.Conv2d(32, 64, 3, 3, 1, 1), nn.ReLU(),
                                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU())

        self.policy_net = nn.Sequential(
            nn.Conv2d(128, 4, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(4 * board_size * board_size, board_size * board_size))

        self.value_net = nn.Sequential(
            nn.Conv2d(128, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 64), nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, state):
        feat = self.feat_net(state)
        prob = self.policy_net(feat)
        val = self.value_net(feat)
        return prob, val

    def evaluate(self, x):
        with torch.no_gard():
            prob, val = self.forward(x)
            return prob.squeeze(), val.squeeze()


class TreeNode(object):
    def __init__(self, parent, prior) -> None:
        self.parent = parent
        self.prior = prior

        self.Q = 0  # 价值函数
        self.N = 0  # 访问次数
        self.children = {}  # a map from action to TreeNode

    def score(self, c_puct):
        # PUCT 分数
        sqrt_sum = np.sqrt(np.sum(node.N for node in self.parent.children))

        return self.Q + c_puct * self.prior * sqrt_sum / (1 + self.N)

    def select(self, c_puct):
        return max(self.children.items(),
                   key=lambda act_node: act_node[1].score(c_puct))

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = TreeNode(self, prior)

    def update(self, qval):

        self.Q = self.Q * self.N + qval
        self.N += 1
        self.Q = self.Q / self.N

    def backup(self, qval):

        self.update(qval)
        if self.parent:
            self.parent.backup(-qval)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0
