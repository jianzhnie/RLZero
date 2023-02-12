import numpy as np


class TreeNode(object):
    def __init__(self, parent, prior) -> None:
        self.parent = parent
        self.prior = prior

        self.Q = 0
        self.U = 0  # score
        self.N = 0
        self.children = {}  # a map from action to TreeNode

    def score(self, c_puct):

        self.U = (c_puct * self.prior * np.sqrt(self.parent.N) / (1 + self.N))

        return self.Q + self.U

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