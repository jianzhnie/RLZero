from operator import itemgetter

import numpy as np

from .mcts import MCTS


class RolloutMCTS(MCTS):

    def __init__(
        self,
        n_playout: int = 1000,
        c_puct: float = 5.0,
        n_limit: int = 1000,
        is_selfplay: bool = False,
    ) -> None:
        MCTS.__init__(self, n_playout, c_puct, is_selfplay=is_selfplay)
        self.n_limit = n_limit

    def _evaluate(self, game_env):
        """Use the rollout policy to play until the end of the game, returning.

        +1 if the current player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        action_probs = self._policy(game_env)
        (
            is_end,
            _,
        ) = game_env.game_end(
        )  # from the perspective of beginning of the rollout

        # begin rollout
        for i in range(self.n_limit):
            rollout_end, rollout_winner = game_env.game_end()
            if rollout_end:
                break
            rollout_action = self._rollout(game_env)
            game_env.step(rollout_action)
        else:
            # If no break from the loop, issue a warning.
            print('WARNING: rollout reached move limit')

        # set leaf_value
        if rollout_winner == -1:  # tie
            leaf_value = 0
        else:
            leaf_value = (1.0 if rollout_winner
                          == game_env.get_current_player() else -1.0)

        return is_end, action_probs, leaf_value

    def _play(self, temperature: float = 1e-3):
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def _rollout(self, game_env):
        """rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        action_probs = np.random.rand(len(game_env.availables))
        tmp_action_probs = zip(game_env.availables, action_probs)
        return max(tmp_action_probs, key=itemgetter(1))[0]

    def _policy(self, game_env):
        """a function that takes in a state and outputs a list of (action,
        probability) tuples."""
        # return uniform probabilities and 0 score for pure MCTS
        action_probs = np.ones(len(game_env.availables)) / len(
            game_env.availables)
        return zip(game_env.availables, action_probs)

    def __str__(self):
        return 'RolloutMCTS'
