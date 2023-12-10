"""Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes.

@author: Junxiao Song
"""

import copy
from typing import Any, Callable, Literal

import numpy as np

from .node import TreeNode


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaZeroMCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(
        self,
        policy_value_fn: Callable,
        c_puct: float = 5,
        n_playout: int = 10000,
        is_selfplay: bool = True,
    ) -> None:
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        # root node do not have parent ,and sure with prior probability 1
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._is_selfplay = is_selfplay

    def _playout(self, game_env) -> None:
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.

        game_env is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            game_env.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(game_env)
        # Check for end of game.
        end, winner = game_env.game_end()
        if not end:
            node.expand(action_probs, add_noise=self._is_selfplay)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game_env.get_current_player(
                ) else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self,
                       game_env,
                       temperature: float = 1e-3) -> tuple[Any, Any]:
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.

        game_env: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            env_copy = copy.deepcopy(game_env)
            self._playout(env_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temperature *
                            np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move: int) -> None:
        """Step forward in the tree, keeping everything we already know about
        the subtree."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self) -> Literal['MCTS']:
        return 'MCTS'
