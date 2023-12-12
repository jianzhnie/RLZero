from typing import List

import numpy as np


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p: float):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0  # 访问次数
        self._Q = 0  # 价值
        self._u = 0  # score u
        self._P = prior_p  # 先验概率
        # its the prior probability that action's taken to get this node

    def select(self, c_puct: float):
        """Select action among children that gives maximum action value Q plus
        bonus u(P).

        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors: List, add_noise: bool = False):
        """Expand tree by creating new children.

        When train by self-play, add dirichlet noises in each node.
        Should note it's different from paper that only add noises in root node,
        I guess alphago zero discard the whole tree after each move and rebuild a new tree,
        so it's no conflict.
        While here i contained the Node under the chosen action, it's a little different.
        There's no idea which is better in addition, the parameters should be tried for 11x11 board.
        Dirichlet parameter: 0.3 is ok, should be smaller with a bigger board, such as 20x20 with 0.03
        weights between priors and noise: 0.75 and 0.25 in paper and i don't change it here,
        But i think maybe 0.8/0.2 or even 0.9/0.1 is better because i add noise in every node
        rich people can try some other parameters

        Args:
            action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        action_priors = list(action_priors)
        if add_noise:
            length = len(action_priors)
            dirichlet_noise = np.random.dirichlet(0.3 * np.ones(length))
            for idx, (action, prob) in enumerate(action_priors):
                if action not in self._children:
                    noise_prob = 0.75 * prob + 0.25 * dirichlet_noise[idx]
                    self._children[action] = TreeNode(self, noise_prob)
        else:
            for action, prob in action_priors:
                if action not in self._children:
                    self._children[action] = TreeNode(self, prob)

    def get_value(self, c_puct: float):
        """Calculate and return the value for this node.

        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) /
                   (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value: float) -> None:
        """Update node values from leaf evaluation.

        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        # 更新访问次数
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # 更新值估计：(v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)
        # This step combine W,Q. Derived formula is as follows:
        # W = W_old + leaf_value;
        # Q_old = W_old / (n-1) => W_old = (n-1)*Q_old;
        # Q = W/n
        # Q = W/n=(W_old + leaf_value)/n = ((n-1)*Q_old+leaf_value)/n
        #   = (n*Q_old-Q_old+leaf_value)/n = Q_old + (leaf_value-Q_old)/n
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value: float) -> None:
        """Like a call to update(), but applied recursively for all
        ancestors."""
        # If it is not root, this node's parent should be updated first.
        # 若该节点不是根节点，则递归更新
        if self._parent:
            # 通过传递取反后的值来改变玩家的视角
            self._parent.update_recursive(-leaf_value)
            # we should change the perspective by the way of taking the negative
        self.update(leaf_value)

    def select_action(self, temperature: float):
        """Select action according to the visit count distribution and the
        temperature."""
        visit_counts = np.array(
            [child._n_visits for child in self._children.values()])
        actions = [action for action in self._children.keys()]
        if temperature == 0:
            action = actions[np.argmax(visit_counts)]
        elif temperature == float('inf'):
            action = np.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts**(1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)

        return action

    def is_leaf(self) -> bool:
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self) -> bool:
        """check if it's root node."""
        return self._parent is None

    def __str__(self) -> str:
        s = []
        s.append(f'Q-Value:  {self._Q}')
        s.append(f'UctScore: {self._u}')
        s.append(f'numVisits: {self._n_visits}')
        return '%s: {%s}' % (self.__class__.__name__, ', '.join(s))
