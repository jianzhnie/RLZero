"""Monte-Carlo Tree Search algorithm for game play."""
import copy
import math
import time
from abc import ABC
from typing import Any, List, Tuple

import numpy as np
from games.base_env import BaseEnv

from .player import Player


class Evaluator(object):
    """Abstract class representing an evaluation function for a game.

    The evaluation function takes in an intermediate state in the game and
    returns an evaluation of that state, which should correlate with chances of
    winning the game. It returns the evaluation from all player's perspectives.
    """

    def evaluate(self, game_env: BaseEnv):
        """Returns evaluation on given state."""
        raise NotImplementedError

    def prior(self, game_env: BaseEnv):
        """Returns a probability for each legal action in the given state."""
        raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
    """A simple evaluator doing random rollouts.

    This evaluator returns the average outcome of playing random actions from
    the given state until the end of the game.  n_rollouts is the number of
    random outcomes to be considered.
    """

    def __init__(self, n_rollouts: int = 20, random_state: Any = None):
        self.n_rollouts = n_rollouts
        self._random_state: np.random.RandomState = (random_state or
                                                     np.random.RandomState())

    def evaluate(self, game_env: BaseEnv):
        """Returns evaluation on given state."""
        result = None
        for _ in range(self.n_rollouts):
            working_env = copy.deepcopy(game_env)
            while not working_env.is_terminal():
                legal_actions = working_env.legal_actions(
                    game_env.current_player())
                action = self._random_state.choice(legal_actions)
                working_env.step(action)
            returns = np.array(working_env.returns())
            result = returns if result is None else result + returns

        return result / self.n_rollouts

    def prior(self, game_env: BaseEnv) -> List[Tuple[int, float]]:
        """Returns equal probability for all actions."""
        legal_actions = game_env.legal_actions(game_env.current_player())
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


class SearchNode(ABC):
    """A node in the search tree.

    This is an abstract base class that outlines the fundamental methods for a
    Monte Carlo Tree node. A SearchNode represents a state and possible continuations
    from it. Each child represents a possible action, and the expected result from doing so.

    Attributes:
      action: The action from the parent node's perspective. Not important for the
        root node, as the actions that lead to it are in the past.
      player: Which player made this action.
      prior: A prior probability for how likely this action will be selected.
      explore_count: How many times this node was explored.
      total_reward: The sum of rewards of rollouts through this node, from the
        parent node's perspective. The average reward of this node is
        `total_reward / explore_count`
      outcome: The rewards for all players if this is a terminal node or the
        subtree has been proven, otherwise None.
      children: A list of SearchNodes representing the possible actions from this
        node, along with their expected rewards.
    """

    __slots__ = [
        'action',
        'player',
        'prior',
        'explore_count',
        'total_reward',
        'outcome',
        'children',
    ]

    def __init__(self, action: int, player: int, prior: float):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

    def uct_value(self, parent_explore_count: int, uct_c: float) -> float:
        """
        Overview: Returns the UCT value of child.

            This function finds the best child node which has the highest UCB (Upper Confidence Bound) score.
            The UCB formula is:
            {UCT}(v_i, v) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\log(N(v))}{N(v_i)}}
                - Q(v_i) is the estimated value of the child node v_i.
                - N(v_i) is a counter of how many times the child node v_i has been on the backpropagation path.
                - N(v) is a counter of how many times the parent (current) node v has been on the backpropagation path.
                - c is a parameter which balances exploration and exploitation.
        Arguments:
            - parent_explore_count (:obj:`int`): the number of times the parent node has been visited.
            - uct_c (:obj:`float`): a parameter which controls the balance between exploration and exploitation. Default value is 1.4.
        """

        if self.outcome is not None:
            return self.outcome[self.player]

        if self.explore_count == 0:
            return float('inf')

        return self.total_reward / self.explore_count + uct_c * math.sqrt(
            math.log(parent_explore_count) / self.explore_count)

    def puct_value(self, parent_explore_count: int, uct_c: float) -> float:
        """
        Overview: Returns the PUCT value of child.

            This function finds the best child node which has the highest UCB (Upper Confidence Bound) score.
            The UCB formula is:
            {UCT}(v_i, v) = \frac{Q(v_i)}{N(v_i)} + c \sqrt{\frac{\log(N(v))}{N(v_i)}}
                - Q(v_i) is the estimated value of the child node v_i.
                - N(v_i) is a counter of how many times the child node v_i has been on the backpropagation path.
                - N(v) is a counter of how many times the parent (current) node v has been on the backpropagation path.
                - c is a parameter which balances exploration and exploitation.
        Arguments:
            - parent_explore_count (:obj:`int`): the number of times the parent node has been visited.
            - uct_c (:obj:`float`): a parameter which controls the balance between exploration and exploitation. Default value is 1.4.
        """
        if self.outcome is not None:
            return self.outcome[self.player]

        return (self.explore_count and self.total_reward / self.explore_count
                ) + uct_c * self.prior * math.sqrt(parent_explore_count) / (
                    self.explore_count + 1)

    def sort_key(self) -> Tuple[float, int, float]:
        """Returns the best action from this node, either proven or most
        visited.

        This ordering leads to choosing:
        - Highest proven score > 0 over anything else, including a promising but
          unproven action.
        - A proven draw only if it has higher exploration than others that are
          uncertain, or the others are losses.
        - Uncertain action with most exploration over loss of any difficulty
        - Hardest loss if everything is a loss
        - Highest expected reward if explore counts are equal (unlikely).
        - Longest win, if multiple are proven (unlikely due to early stopping).
        """
        return (
            0 if self.outcome is None else self.outcome[self.player],
            self.explore_count,
            self.total_reward,
        )

    def best_child(self) -> 'SearchNode':
        """Returns the best child in order of the sort key."""
        return max(self.children, key=SearchNode.sort_key)

    def rollout_policy(self, possible_actions: List[int]) -> int:
        """
        Overview:
            This method implements the rollout policy for a node during the Monte Carlo Tree Search.
            The rollout policy is used to determine the action taken during the simulation phase of the MCTS.
            In this case, the policy is to randomly choose an action from the list of possible actions.
        Arguments:
            - possible_actions(:obj:`list`): A list of all possible actions that can be taken from the current state.
        Return:
            - action(:obj:`int`): A randomly chosen action from the list of possible actions.
        """
        return possible_actions[np.random.randint(len(possible_actions))]

    def children_str(self, game_env: BaseEnv = None) -> str:
        """Returns the string representation of this node's children.

        They are ordered based on the sort key, so order of being chosen to play.

        Args:
          state: A `pyspiel.State` object, to be used to convert the action id into
            a human readable format. If None, the action integer id is used.
        """
        return '\n'.join([
            c.to_str(game_env)
            for c in reversed(sorted(self.children, key=SearchNode.sort_key))
        ])

    def to_str(self, game_env: BaseEnv = None) -> str:
        """Returns the string representation of this node.

        Args:
          state: A `pyspiel.State` object, to be used to convert the action id into
            a human readable format. If None, the action integer id is used.
        """
        action = str(self.action)
        return (
            'action:{}, player: {}, prior: {:5.3f}, value: {:6.3f}, sims: {:5d}, '
            'outcome: {}, {:3d} children').format(
                action,
                self.player,
                self.prior,
                self.explore_count and self.total_reward / self.explore_count,
                self.explore_count,
                ('{:4.1f}'.format(self.outcome[self.player])
                 if self.outcome else 'none'),
                len(self.children),
            )

    def __str__(self) -> str:
        return self.to_str(None)


class MCTSNode(SearchNode):
    """A node in the search tree.

    Args:
        SearchNode (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
        parent: SearchNode,
        action: int,
        player: int,
        prior: float,
    ):
        super().__init__(action, player, prior)
        self.parent = parent
        self.action = action
        self.children = []
        self.player = player

    def select(self) -> 'SearchNode':
        best_child = self.best_child()
        best_action = best_child.action
        return best_action

    def expand(
        self,
        player: int,
        action_priors: List[Tuple[int, float]],
    ) -> None:
        """
        Overview:
            This method expands the current node by creating a new child node.
            It pops an action from the list of legal actions, simulates the action to get the next game state,
            and creates a new child node with that state. The new child node is then added to the list of children nodes.
        Args:
            player (int): The player who is going to play the action.
            action_priors (List[Tuple[int, float]]): A list of tuples containing the action and the prior probability of the action.
        Returns:
            - node(:obj:`TwoPlayersMCTSNode`): The child node object that has been created.
        """
        action_priors = list(action_priors)
        for action, prior in action_priors:
            child_node = MCTSNode(parent=self,
                                  action=action,
                                  player=player,
                                  prior=prior)
            self.children.append(child_node)

        return

    def backpropagate(self, reward: float) -> None:
        """
        Overview:
            This method performs backpropagation from the current node.
            It increments the number of times the node has been visited and the number of wins for the result.
            If the current node has a parent, the method recursively backpropagates the result to the parent.
        """
        self.explore_count += 1.0
        # result is the index of the self._results list.
        # result = ±1 when player 1 wins/loses the game.
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def _tree_policy(self, root: SearchNode, game_env: BaseEnv):
        """
        Overview:
            This function implements the tree search from the root node to the leaf node, which is either visited for the first time or is the terminal node.
            At each step, if the current node is not fully expanded, it expands.
            If it is fully expanded, it moves to the best child according to the tree policy.
        Returns:
            - node(:obj:`TwoPlayersMCTSNode`): The leaf node object that has been reached by the tree search.
        """
        current_node: SearchNode = SearchNode(None,
                                              game_env.current_player(),
                                              prior=1.0)
        working_env = copy.deepcopy(game_env)
        while not working_env.is_terminal() and current_node.explore_count > 0:
            if not current_node.children:
                # choose a child node which has not been visited before
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def rollout(self, game_env: BaseEnv):
        """
        Overview:
            This method performs a rollout (simulation) from the current node.
            It repeatedly selects an action based on the rollout policy and simulates the action until the game ends.
            The method then returns the reward of the game's final state.
        Returns:
            -reward (:obj:`int`): reward = ±1 when player 1 wins/loses the game, reward = 0 when it is a tie, reward = None when current node is not a terminal node.
        """
        current_rollout_env = copy.deepcopy(game_env)
        while not current_rollout_env.is_terminal():
            possible_actions = current_rollout_env.legal_actions(
                current_rollout_env.current_player())
            action = self.rollout_policy(possible_actions)
            current_rollout_env = current_rollout_env.step(action)
        return current_rollout_env.returns()[self.player]

    def mcts_simulate(
        self,
        game_env: BaseEnv,
        max_simulation_number: int = 1000,
        max_simulation_seconds: int = 100,
    ) -> None:
        """_summary_

        Args:
            game_env (BaseEnv): _description_
        """
        if max_simulation_number is None and max_simulation_seconds is None:
            raise ValueError(
                'Either max_simulation_number or max_simulation_seconds must be set.'
            )
        if max_simulation_number is None:
            assert max_simulation_seconds is not None
            end_time = time.time() + max_simulation_seconds
            while True:
                if time.time() > end_time:
                    break

    def is_fully_expanded(self):
        """
        Overview:
            This method checks if the node is fully expanded.
            A node is considered fully expanded when all of its child nodes have been selected at least once.
            Whenever a new child node is selected for the first time, a corresponding action is removed from the list of legal actions.
            Once the list of legal actions is depleted, it signifies that all child nodes have been selected,
            thereby indicating that the parent node is fully expanded.
        """
        return len(self.legal_actions()) == 0

    def is_terminal_node(self):
        """
        Overview:
            This function checks whether the current node is a terminal node.
            It uses the game environment's get_done_reward method to check if the game has ended.
        Returns:
            - A bool flag representing whether the game is over.
        """
        # The get_done_reward() returns a tuple (done, reward).
        # reward = ±1 when player 1 wins/loses the game.
        # reward = 0 when it is a tie.
        # reward = None when current node is not a terminal node.
        # done is the bool flag representing whether the game is over.
        return self.env.get_done_reward()[0]


class DeepMindMCTS:
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(
        self,
        game_env: None,
        uct_c: float = 2,
        max_simulations: int = 2000,
        evaluator: RandomRolloutEvaluator = RandomRolloutEvaluator(),
        child_selection_method: str = 'puct',
        add_exploration_noise: bool = False,
        dirichlet_noise_alpha: float = 1.0,
        dirichlet_noise_epsilon: float = 0.25,
        solve: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initializes a MCTS Search algorithm in the form of a bot.

        In multiplayer games, or non-zero-sum games, the players will play the
        greedy strategy.

        Args:
          game_env: A Game Env to play.
          uct_c: The exploration constant for UCT.
          max_simulations: How many iterations of MCTS to perform. Each simulation
            will result in one call to the evaluator. Memory usage should grow
            linearly with simulations * branching factor. How many nodes in the
            search tree should be evaluated. This is correlated with memory size and
            tree depth.
          evaluator: A `Evaluator` object to use to evaluate a leaf node.
          solve: Whether to back up solved states.
          child_selection_fn: A function to select the child in the descent phase.
            The default is UCT.
          dirichlet_noise: A tuple of (epsilon, alpha) for adding dirichlet noise to
            the policy at the root. This is from the alpha-zero paper.
          verbose: Whether to print information about the search tree before
            returning the action. Useful for confirming the search is working
            sensibly.

        Raises:
          ValueError: if the game type isn't supported.
        """
        self.game_env = game_env
        self.uct_c = uct_c
        self.max_simulations = max_simulations
        self.evaluator = evaluator
        if child_selection_method == 'puct':
            self._child_selection_fn = SearchNode.puct_value
        elif child_selection_method == 'uct':
            self._child_selection_fn = SearchNode.uct_value
        self.max_utility = game_env.max_utility()
        if add_exploration_noise:
            assert dirichlet_noise_alpha is not None
            assert dirichlet_noise_epsilon is not None
        self.add_exploration_noise = add_exploration_noise
        self.dirichlet_noide_alpha = dirichlet_noise_epsilon
        self.dirichlet_noise_epsilon = dirichlet_noise_epsilon
        self.verbose = verbose
        self.solve = solve
        self._random_state = np.random.RandomState()

    def step_with_policy(
            self, game_env: BaseEnv) -> Tuple[List[Tuple[int, float]], int]:
        """Returns bot's policy and action at given state."""
        t1 = time.time()
        root: SearchNode = self.mcts_search(game_env)
        best: SearchNode = root.best_child()

        if self.verbose:
            seconds = time.time() - t1
            print('Finished {} sims in {:.3f} secs, {:.1f} sims/s'.format(
                root.explore_count, seconds, root.explore_count / seconds))
            print('Root:')
            print(root.to_str(game_env))
            print('Children:')
            print(root.children_str(game_env))
            if best.children:
                working_env = copy.deepcopy(game_env)
                working_env.step(best.action)
                print('Children of chosen:')
                print(best.children_str(working_env))

        mcts_action = best.action

        legal_actions = game_env.legal_actions(game_env.current_player())
        policy = [(action, (1.0 if action == mcts_action else 0.0))
                  for action in legal_actions]

        return policy, mcts_action

    def step(self, game_env: BaseEnv) -> int:
        return self.step_with_policy(game_env)[1]

    def _apply_tree_policy(
        self,
        root: SearchNode,
        game_env: BaseEnv,
    ) -> Tuple[List[SearchNode], BaseEnv]:
        """Applies the UCT policy to play the game until reaching a leaf node.

        A leaf node is defined as a node that is terminal or has not been evaluated
        yet. If it reaches a node that has been evaluated before but hasn't been
        expanded, then expand it's children and continue.

        Args:
          root: The root node in the search tree.
          state: The state of the game at the root node.

        Returns:
          visit_path: A list of nodes descending from the root node to a leaf node.
          working_state: The state of the game at the leaf node.
        """
        visit_path = [root]
        print('game_env: ')
        print(game_env.legal_actions())
        working_env = copy.deepcopy(game_env)
        print('working_env: ')
        print(working_env.legal_actions())
        current_node: SearchNode = root
        while not working_env.is_terminal() and current_node.explore_count > 0:
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_env)
                if current_node is root and self.add_exploration_noise:
                    legal_actions = self._add_dirichlet_noise(legal_actions)
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_env.current_player()
                current_node.children = [
                    SearchNode(action, player, prior)
                    for action, prior in legal_actions
                ]

            # Otherwise choose node with largest UCT value
            chosen_child: SearchNode = max(
                current_node.children,
                key=lambda c: self._child_selection_fn(
                    c, current_node.explore_count, self.uct_c),
            )

            working_env.step(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_env

    def _add_dirichlet_noise(
            self,
            legal_actions: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        legal_actions = list(legal_actions)
        # Compute the weight of the exploration noise.
        epsilon = self.dirichlet_noise_epsilon
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self.dirichlet_noide_alpha] * len(legal_actions)
        # Generate Dirichlet noise using the alpha values.
        dirichlet_noise = self._random_state.dirichlet(alpha)
        noise_action_policy = []
        # Update the prior probability of each child node with the exploration noise.
        for (action, prob), noise in zip(legal_actions, dirichlet_noise):
            noise_prob = epsilon * noise + (1 - epsilon) * prob
            noise_action_policy.append((action, noise_prob))
        return noise_action_policy

    def mcts_search(self, game_env: BaseEnv) -> SearchNode:
        """A vanilla Monte-Carlo Tree Search algorithm.

        This algorithm searches the game tree from the given state.
        At the leaf, the evaluator is called if the game state is not terminal.
        A total of max_simulations states are explored.

        At every node, the algorithm chooses the action with the highest PUCT value,
        defined as: `Q/N + c * prior * sqrt(parent_N) / N`, where Q is the total
        reward after the action, and N is the number of times the action was
        explored in this position. The input parameter c controls the balance
        between exploration and exploitation; higher values of c encourage
        exploration of under-explored nodes. Unseen actions are always explored
        first.

        At the end of the search, the chosen action is the action that has been
        explored most often. This is the action that is returned.

        This implementation supports sequential n-player games, with or without
        chance nodes. All players maximize their own reward and ignore the other
        players' rewards. This corresponds to max^n for n-player games. It is the
        norm for zero-sum games, but doesn't have any special handling for
        non-zero-sum games. It doesn't have any special handling for imperfect
        information games.

        The implementation also supports backing up solved states, i.e. MCTS-Solver.
        The implementation is general in that it is based on a max^n backup (each
        player greedily chooses their maximum among proven children values, or there
        exists one child whose proven value is game.max_utility()), so it will work
        for multiplayer, general-sum, and arbitrary payoff games (not just win/loss/
        draw games). Also chance nodes are considered proven only if all children
        have the same value.

        Some references:
        - Sturtevant, An Analysis of UCT in Multi-Player Games,  2008,
          https://web.cs.du.edu/~sturtevant/papers/multi-player_UCT.pdf
        - Nijssen, Monte-Carlo Tree Search for Multi-Player Games, 2013,
          https://project.dke.maastrichtuniversity.nl/games/files/phd/Nijssen_thesis.pdf
        - Silver, AlphaGo Zero: Starting from scratch, 2017
          https://deepmind.com/blog/article/alphago-zero-starting-scratch
        - Winands, Bjornsson, and Saito, "Monte-Carlo Tree Search Solver", 2008.
          https://dke.maastrichtuniversity.nl/m.winands/documents/uctloa.pdf

        Arguments:
          game_env: game env state to search from

        Returns:
          The most visited move from the root node.
        """
        root: SearchNode = SearchNode(action=None,
                                      player=game_env.current_player(),
                                      prior=1)
        for _ in range(self.max_simulations):
            visit_path, working_env = self._apply_tree_policy(root, game_env)
            if working_env.is_terminal():
                returns = working_env.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_env)
                solved = False

            while visit_path:
                decision_node_idx = -1
                # Chance node targets are for the respective decision-maker.
                target_return = returns[visit_path[decision_node_idx].player]
                node: SearchNode = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    # 这里，node 为非叶子节点，node.children 为 node 的所有子节点
                    # node.children[0].player 为 node 的第一个子节点的 player
                    # If any have max utility (won?), or all children are solved,
                    # choose the one best for the player choosing.
                    best: SearchNode = None
                    all_solved = True
                    for child in node.children:
                        if child.outcome is None:
                            all_solved = False
                        elif (best is None
                              or child.outcome[player] > best.outcome[player]):
                            best = child
                    if best is not None and (all_solved or best.outcome[player]
                                             == self.max_utility):
                        node.outcome = best.outcome
                    else:
                        solved = False
            if root.outcome is not None:
                break

        return root


class MCTSBot(Player):

    def __init__(self,
                 game_env: BaseEnv,
                 max_simulations=1000,
                 player_id: int = 0,
                 player_name: str = '') -> None:
        super().__init__(player_id, player_name)

        evaluator = RandomRolloutEvaluator(n_rollouts=20)
        self.mcts: DeepMindMCTS = DeepMindMCTS(
            game_env,
            uct_c=2,
            max_simulations=max_simulations,
            evaluator=evaluator,
            child_selection_method='puct',
            add_exploration_noise=True,
            dirichlet_noise_alpha=1.0,
            dirichlet_noise_epsilon=0.25,
            solve=False,
            verbose=False,
        )

    def set_player_id(self, player_id):
        self.player_id = player_id

    def get_player_id(self):
        return self.player_id

    def get_player_name(self):
        return self.player_name

    def get_action(self, game_env: BaseEnv, **kwargs):
        sensible_moves = game_env.leagel_actions()
        if len(sensible_moves) > 0:
            action = self.mcts.step(game_env)
            return action
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'DeepMindMCTSBot, id: {}, name: {}.'.format(
            self.get_player_id(), self.get_player_name())
