import copy
from typing import Callable

import numpy as np

from .node import TreeNode
from .player import Player


def softmax(x):
    """avoid data overflow."""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaZeroMCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(
        self,
        policy_value_fn: Callable,
        n_playout: int = 10000,
        c_puct: float = 5,
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
        self.policy_value_fn = policy_value_fn
        self.n_playout = n_playout
        self._c_puct = c_puct
        self._is_selfplay = is_selfplay

    def _playout(self, game_env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.

        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while 1:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            game_env.step(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self.policy_value_fn(game_env)
        is_end, winner = game_env.game_end()
        if not is_end:
            node.expand(action_probs)
        else:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game_env.get_current_player(
                ) else -1.0

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def simulate(self, game_env, temperature: float = 1e-3):
        """Runs all simulations sequentially and returns the available actions and their corresponding probabilities
        Arguments:
        state -- the current state, including both game state and the current player.
        temperature -- temperature parameter in (0, 1] that controls the level of exploration
        Returns:
        the available actions and the corresponding probabilities
        """
        # The slowest section!!!! how to speed up!!
        for n in range(self.n_playout):
            env_copy = copy.deepcopy(game_env)
            # key!!!, can't change the state object
            self._playout(env_copy)  # the state_copy reference will be changed

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temperature *
                            np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know about
        the subtree."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'AlphaZeroMCTS'


class AlphaZeroPlayer(Player):
    """AI player based on MCTS."""

    def __init__(
        self,
        policy_value_fn,
        n_playout: int = 2000,
        c_puct: float = 5,
        is_selfplay: bool = False,
        add_noise: bool = False,
    ) -> None:
        self.mcts = AlphaZeroMCTS(
            policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=is_selfplay,
        )
        self.is_selfplay = is_selfplay
        self._add_noise = is_selfplay if add_noise is None else add_noise

    def reset_player(self):
        """reset, reconstructing the MCTS Tree for next simulation."""
        self.mcts.update_with_move(-1)

    def get_action(
        self,
        game_env,
        temperature: float = 1e-3,
        return_prob: bool = False,
    ):
        sensible_moves = game_env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(game_env.width * game_env.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.simulate(game_env, temperature)
            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            if self.is_selfplay:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'AlphaZeroPlayer, id: {}, name: {}.'.format(
            self.get_player_id(), self.get_player_name())
