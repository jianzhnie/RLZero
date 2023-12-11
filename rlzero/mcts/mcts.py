import copy

from .node import TreeNode


class MCTS(object):

    def __init__(
        self,
        n_playout: int = 1000,
        c_puct: float = 5.0,
        is_selfplay: bool = False,
    ) -> None:
        self._root = TreeNode(parent=None, prior_p=1.0)
        self.n_playout = n_playout  # number of plays of one simulation
        self._c_puct = c_puct
        # a number controlling the relative impact of values, Q, and P
        self._is_selfplay = is_selfplay  # whether used to selfplay

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

        # Evaluate the leaf using a network which outputs a list of (action, probability)
        # tuples p and also a score v in [-1, 1] for the current player.
        is_end, action_prob, leaf_value = self._evaluate(game_env)
        # MCTS Of the EVALUATE step

        if not is_end:
            node.expand(action_prob, add_noise=self._is_selfplay)
        # MCTS of the [EXPAND] step

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)  # MCTS of the [BACKUP] step
        # print('after update...', node._n_visits, node._Q)

    def _evaluate(self, game_env):
        """Template Method, Override for different child class MCTS of the.

        [EVALUATE] Step Return the move probabilities of each available action and the evaluation value of winning.
        """
        raise NotImplementedError

    def _play(self, temperature=1e-3):
        """Template Method, Override for different child class MCTS of the.

        [PLAY] Step Return the final action.
        """
        raise NotImplementedError

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

        return self._play(temperature)  # override for different child class

    def __str__(self):
        return 'MCTS'
