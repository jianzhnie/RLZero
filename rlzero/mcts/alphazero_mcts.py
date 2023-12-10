from typing import Any, Callable, Literal

import numpy as np

from .mcts import MCTS


def softmax(x):
    """avoid data overflow."""
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class AlphaZeroMCTS(MCTS):
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
        MCTS.__init__(self, n_playout, c_puct, is_selfplay=is_selfplay)
        self._policy = policy_value_fn
        self._is_selfplay = is_selfplay

    def _evaluate(self, game_env):
        action_probs, leaf_value = self._policy(game_env)

        # Check for end of game, Adjust the leaf_value
        # if end, then policy evaluation
        is_end, winner = game_env.game_end()
        if is_end:
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == game_env.get_current_player(
                ) else -1.0

        return is_end, action_probs, leaf_value

    def _play(self, temperature: float = 1e-3) -> tuple[Any, Any]:
        """
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temperature *
                            np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def __str__(self) -> Literal['MCTS']:
        return 'MCTS'
