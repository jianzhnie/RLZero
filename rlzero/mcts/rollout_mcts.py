import copy
from operator import itemgetter

import numpy as np

from .node import TreeNode
from .player import Player


class RolloutMCTS(object):

    def __init__(
        self,
        n_playout: int = 1000,
        c_puct: float = 5.0,
        n_limit: int = 1000,
    ) -> None:
        self._root = TreeNode(parent=None, prior=1.0)
        self.n_playout = n_playout
        self._c_puct = c_puct
        self.n_limit = n_limit

    def _playout(self, game_env):
        """Run a single search from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents."""
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # MCTS of SELECT step
            game_env.step(action)
            # print('select action is ...',action)
            # print(action, game_env.availables)

        action_probs = self.policy_value_fn(game_env)
        # print('action_probs is ...', action_probs)
        # Check for end of game
        is_end, _ = game_env.game_end()
        if not is_end:
            node.expand(action_probs)
        # MCTS of the [EXPAND] step
        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        leaf_value = self._evaluate(game_env)
        # MCTS Of the EVALUATE step
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        # MCTS of the [BACKUP] step
        # print('after update...', node.visit_count, node.total_reward)

    def _evaluate(self, game_env):
        """Use the rollout policy to play until the end of the game, returning.

        +1 if the current player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        # begin rollout
        for i in range(self.n_limit):
            rollout_end, rollout_winner = game_env.game_end()
            if rollout_end:
                break
            rollout_probs = self.rollout_policy(game_env)
            rollout_action = max(rollout_probs, key=itemgetter(1))[0]
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

        return leaf_value

    def simulate(self, game_env, temperature: float = 0.001):
        for n in range(self.n_playout):
            env_copy = copy.deepcopy(game_env)
            self._playout(env_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1].visit_count)[0]

    def update_with_move(self, last_move: int):
        """Step forward in the tree, keeping everything we already know about
        the subtree.

        if self-play then update the root node and reuse the search tree, speeding next simulation else reset the root
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def rollout_policy(self, game_env):
        """rollout_policy_fn -- a coarse, fast version of policy_fn used in the rollout phase."""
        # rollout randomly
        action_probs = np.random.rand(len(game_env.availables))
        return zip(game_env.availables, action_probs)

    def policy_value_fn(self, game_env):
        """a function that takes in a state and outputs a list of (action,
        probability) tuples."""
        # return uniform probabilities and 0 score for pure MCTS
        action_probs = np.ones(len(game_env.availables)) / len(
            game_env.availables)
        return zip(game_env.availables, action_probs)

    def __str__(self):
        return 'RolloutMCTS'


class RolloutPlayer(Player):

    def __init__(
        self,
        n_playout: int = 1000,
        c_puct: float = 5,
        player_id: int = 0,
        player_name: str = '',
    ) -> None:
        super().__init__(player_id, player_name)
        self.mcts = RolloutMCTS(n_playout, c_puct)

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, game_env, **kwargs):
        sensible_moves = game_env.availables
        if len(sensible_moves) > 0:
            move = self.mcts.simulate(game_env)
            self.mcts.update_with_move(-1)
            return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'RolloutPlayer, id: {}, name: {}.'.format(
            self.get_player_id(), self.get_player_name())
