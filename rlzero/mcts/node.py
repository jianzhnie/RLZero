import math
from typing import List

import numpy as np


class TreeNode(object):
    """A node in the MCTS tree.

    Overview:
        A class for a node in a Monte Carlo Tree. The properties of this class store basic information about the node,
        such as its parent node, child nodes, and the number of times the node has been visited.
        The methods of this class implement basic functionalities that a node should have, such as propagating the value back,
        checking if the node is the root node, and determining if it is a leaf node.
    """

    def __init__(self, parent: 'TreeNode' = None, prior: float = 1.0) -> None:
        """
        Overview:
            Initialize a Node object.
        Arguments:
            - parent (:obj:`Node`): The parent node of the current node.
            - prior (:obj:`Float`): The prior probability of selecting this node.
        """
        # The parent node.
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self.explore_count = 0  # 访问次数
        self.total_reward = 0  # 价值
        self.prior = prior  # 先验概率

    def select(self, c_puct: float):
        """Select action among children that gives maximum action value Q plus
        bonus u(P).

        Return: A tuple of (action, next_node)
        """
        if not self._children:
            raise ValueError('Node has no children.')

        return max(self._children.items(),
                   key=lambda act_node: act_node[1].uct_value(c_puct))

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

    def uct_value(self, c_puct: float):
        """Returns the UCT value of child."""
        if self._parent.explore_count == 0:
            return float('inf')

        if self.explore_count == 0:
            return float('inf')

        exploration_score = self.total_reward / self.explore_count
        exploitation_score = math.sqrt(
            math.log(self._parent.explore_count) / self.explore_count)

        score = exploration_score + c_puct * exploitation_score
        return score

    def puct_value(self, c_puct: float):
        """Calculate and return the PUCT value for this node.

        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        exploration_score = self.total_reward / self.explore_count
        exploitation_score = self.prior * math.sqrt(
            self._parent.explore_count) / (self.explore_count + 1)
        score = exploration_score + c_puct * exploitation_score
        return score

    def update(self, value: float) -> None:
        """Update the current node information from leaf evaluation, such as
        ``explore_count`` and ``_value_sum``.

        Overview:
            Update the current node information, such as ``explore_count`` and ``_value_sum``.
        Arguments:
            - value (:obj:`Float`): The the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        # 更新访问次数
        self.explore_count += 1
        # Update Q, a running average of values for all visits.
        self.total_reward += value

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

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not, (i.e. no nodes below this have been expanded).
        Returns:
            - output (:obj:`Bool`): If self._children is empty, it means that the node has not
            been expanded yet, which indicates that the node is a leaf node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): If the node does not have a parent node,
            then it is a root node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        """
        Overview:
            Get the parent node of the current node.
        Returns:
            - output (:obj:`Node`): The parent node of the current node.
        """
        return self._parent

    @property
    def children(self) -> None:
        """
        Overview:
            Get the dictionary of children nodes of the current node.
        Returns:
            - output (:obj:`dict`): A dictionary representing the children of the current node.
        """
        return self._children

    def __str__(self) -> str:
        s = ['MCTSNode']
        s.append(f'Total Value:  {self.total_reward}')
        s.append(f'Num Visits: {self.explore_count}')
        return '%s: {%s}' % (self.__class__.__name__, ', '.join(s))
