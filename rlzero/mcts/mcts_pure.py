import copy
from operator import itemgetter
from typing import Any, Literal

import numpy as np


def rollout_policy_fn(game_env) -> zip[tuple[Any, Any]]:
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    action_probs = np.random.rand(len(game_env.availables))
    return zip(game_env.availables, action_probs)


def policy_value_fn(game_env) -> tuple[zip[tuple[Any, Any]], Literal[0]]:
    """a function that takes in a state and outputs a list of (action,
    probability) tuples and a score for the state."""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(game_env.availables)) / len(game_env.availables)
    return zip(game_env.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p: float) -> None:
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0  # 访问次数
        self._Q = 0  # 价值
        self._u = 0  # score u
        self._P = prior_p  # 先验概率

    def select(self, c_puct: float) -> tuple:
        """Select action among children that gives maximum action value Q plus
        bonus u(P).

        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors: list[tuple[Any, Any]]) -> None:
        """Expand tree by creating new children.

        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def update(self, leaf_value: float) -> None:
        """Update node values from leaf evaluation.

        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        # there is just: (v-Q)/(n+1)+Q = (v-Q+(n+1)*Q)/(n+1)=(v+n*Q)/(n+1)
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value: float) -> None:
        """Like a call to update(), but applied recursively for all
        ancestors."""
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct: float) -> Any:
        """Calculate and return the value for this node.

        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) /
                   (1 + self._n_visits))
        return self._Q + self._u

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
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(
        self,
        policy_value_fn,
        c_puct: float = 5,
        n_playout: int = 10000,
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
        self._root = TreeNode(parent=None, prior_p=1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, game_env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.

        game_env is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                # break if the node is leaf node
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            # print('select action is ...',action)
            # print(action,state.availables)
            game_env.do_move(action)

        action_probs, _ = self._policy(game_env)
        # Check for end of game
        end, winner = game_env.game_end()
        if not end:
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(game_env)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)
        print('after update...', node._n_visits, node._Q)

    def _evaluate_rollout(self, game_env, limit: int = 1000):
        """Use the rollout policy to play until the end of the game, returning.

        +1 if the current player wins, -1 if the opponent wins, and 0 if it is a tie.
        """
        player = game_env.get_current_player()
        for i in range(limit):
            end, winner = game_env.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(game_env)
            max_action = max(action_probs, key=itemgetter(1))[0]
            game_env.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print('WARNING: rollout reached move limit')
        # print('winner is ...',winner)
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, game_env) -> Any:
        """Runs all playouts sequentially and returns the most visited action.
        game_env: the current game_env

        Return: the selected action
        """
        for n in range(self._n_playout):
            env_copy = copy.deepcopy(game_env)
            self._playout(env_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move) -> None:
        """Step forward in the tree, keeping everything we already know about
        the subtree."""
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


class MCTSPlayer(object):
    """AI player based on MCTS."""

    def __init__(self, c_puct: float = 5, n_playout: int = 2000) -> None:
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, player_id) -> None:
        self.player = player_id

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def get_action(self, game_env):
        sensible_moves = game_env.availables
        if game_env.last_move != -1:
            self.mcts.update_with_move(last_move=game_env.last_move)
            # reuse the tree
            # retain the tree that can continue to use
            # so update the tree with opponent's move and do mcts from the current node

        if len(sensible_moves) > 0:
            move = self.mcts.get_move(game_env)
            self.mcts.update_with_move(move)
            return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'MCTS {}'.format(self.player)
