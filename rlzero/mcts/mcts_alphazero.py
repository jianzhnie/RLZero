"""Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes.

@author: Junxiao Song
"""

import copy
from typing import Any, Callable, List, Literal, Tuple

import numpy as np


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p: float):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
        # its the prior probability that action's taken to get this node

    def select(self, c_puct: float) -> Tuple(int, Any):
        """Select action among children that gives maximum action value Q plus
        bonus u(P).

        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def expand(self, action_priors: List[int, float], add_noise: bool = False):
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

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        """check if it's root node."""
        return self._parent is None


class MCTS(object):
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

    def _playout(self, game_env):
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


class MCTSPlayer(object):
    """AI player based on MCTS."""

    def __init__(
        self,
        policy_value_function,
        c_puct: float = 5,
        n_playout: int = 2000,
        is_selfplay: bool = False,
    ) -> None:
        self.mcts = MCTS(
            policy_value_function,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=is_selfplay,
        )
        self.is_selfplay = is_selfplay

    def set_player_ind(self, player_id: int) -> None:
        """set player index."""
        self.player = player_id

    def reset_player(self):
        """reset player."""
        self.mcts.update_with_move(-1)

    def get_action(
        self,
        game_env,
        temperature: float = 1e-3,
        return_prob: bool = False,
    ) -> tuple[Any, np.NDArray]:
        sensible_moves = game_env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(game_env.width * game_env.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(game_env, temperature)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                move = np.random.choice(acts, p=probs)
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'MCTS-AlphaZero {}'.format(self.player)
