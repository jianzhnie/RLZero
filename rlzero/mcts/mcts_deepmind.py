"""Monte-Carlo Tree Search algorithm for game play."""

import math
from typing import Any, List

import numpy as np

from games.base_env import BaseEnv


class Evaluator(object):
    """Abstract class representing an evaluation function for a game.

    The evaluation function takes in an intermediate state in the game and
    returns an evaluation of that state, which should correlate with chances of
    winning the game. It returns the evaluation from all player's perspectives.
    """

    def evaluate(self, state):
        """Returns evaluation on given state."""
        raise NotImplementedError

    def prior(self, state):
        """Returns a probability for each legal action in the given state."""
        raise NotImplementedError


class RandomRolloutEvaluator(Evaluator):
    """A simple evaluator doing random rollouts.

    This evaluator returns the average outcome of playing random actions from
    the given state until the end of the game.  n_rollouts is the number of
    random outcomes to be considered.
    """

    def __init__(self, n_rollouts: int = 1, random_state: Any = None):
        self.n_rollouts = n_rollouts
        self._random_state = random_state or np.random.RandomState()

    def evaluate(self, game_env: BaseEnv):
        """Returns evaluation on given state."""
        result = None
        for _ in range(self.n_rollouts):
            working_state = game_env.clone()
            while not working_state.is_terminal():
                action = self._random_state.choice(
                    working_state.legal_actions(game_env.current_player()))
                working_state.step(action)
            returns = np.array(working_state.returns())
            result = returns if result is None else result + returns

        return result / self.n_rollouts

    def prior(self, game_env: BaseEnv):
        """Returns equal probability for all actions."""
        legal_actions = game_env.legal_actions(game_env.current_player())
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]


class SearchNode(object):
    """A node in the search tree.

    A SearchNode represents a state and possible continuations from it. Each child
    represents a possible action, and the expected result from doing so.

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

    def __init__(self, action, player, prior):
        self.action = action
        self.player = player
        self.prior = prior
        self.explore_count = 0
        self.total_reward = 0.0
        self.outcome = None
        self.children = []

    def uct_value(self, parent_explore_count, uct_c):
        """Returns the UCT value of child."""
        if self.outcome is not None:
            return self.outcome[self.player]

        if self.explore_count == 0:
            return float('inf')

        return self.total_reward / self.explore_count + uct_c * math.sqrt(
            math.log(parent_explore_count) / self.explore_count)

    def puct_value(self, parent_explore_count, uct_c):
        """Returns the PUCT value of child."""
        if self.outcome is not None:
            return self.outcome[self.player]

        return (self.explore_count and self.total_reward / self.explore_count
                ) + uct_c * self.prior * math.sqrt(parent_explore_count) / (
                    self.explore_count + 1)

    def sort_key(self):
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

    def best_child(self):
        """Returns the best child in order of the sort key."""
        return max(self.children, key=SearchNode.sort_key)


class MCTSBot:
    """Bot that uses Monte-Carlo Tree Search algorithm."""

    def __init__(
        self,
        game_env: None,
        uct_c: float = 2,
        max_simulations: int = 800,
        evaluator: RandomRolloutEvaluator = None,
        child_selection_method: str = 'puct',
        add_exploration_noise: bool = False,
        dirichlet_alpha: float = None,
        dirichlet_epsilon: float = None,
        solve: bool = True,
        verbose: bool = False,
        dont_return_chance_node: bool = False,
    ):
        """Initializes a MCTS Search algorithm in the form of a bot.

        In multiplayer games, or non-zero-sum games, the players will play the
        greedy strategy.

        Args:
          game: A pyspiel.Game to play.
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
          dont_return_chance_node: If true, do not stop expanding at chance nodes.
            Enabled for AlphaZero.

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
        self.add_exploration_noise = add_exploration_noise
        self.dirichlet_noide_alpha = dirichlet_alpha
        self.dirichlet_noise_epsilon = dirichlet_epsilon
        self.verbose = verbose
        self.solve = solve
        self._random_state = np.random.RandomState()
        self.dont_return_chance_node = dont_return_chance_node

    def step_with_policy(self, state):
        """Returns bot's policy and action at given state."""
        root: SearchNode = self.mcts_search(state)
        best: SearchNode = root.best_child()
        mcts_action = best.action
        policy = [(action, (1.0 if action == mcts_action else 0.0))
                  for action in state.legal_actions(state.current_player())]
        return policy, mcts_action

    def step(self, state):
        return self.step_with_policy(state)[1]

    def _apply_tree_policy(self, root: SearchNode, game_env: BaseEnv):
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
        working_state_env = game_env.clone()
        current_node: SearchNode = root
        while not working_state_env.is_terminal(
        ) and current_node.explore_count > 0:
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_state_env)
                if current_node is root and self.add_exploration_noise:
                    epsilon, alpha = self.dirichlet_noise_epsilon, self.dirichlet_noide_alpha
                    noise = self._random_state.dirichlet([alpha] *
                                                         len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                                     for (a,
                                          p), n in zip(legal_actions, noise)]
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_state_env.current_player()
                current_node.children = [
                    SearchNode(action, player, prior)
                    for action, prior in legal_actions
                ]

            # Otherwise choose node with largest UCT value
            chosen_child: SearchNode = max(
                current_node.children,
                key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                    c, current_node.explore_count, self.uct_c),
            )

            working_state_env.step(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state_env

    def _add_exploration_noise(self, leagel_actions: List[int]) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(leagel_actions)
        epsilon = self.dirichlet_noise_epsilon
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self.dirichlet_noide_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        dirichlet_noise = np.random.dirichlet(alpha)
        noise_action_probs = []
        # Update the prior probability of each child node with the exploration noise.
        for (action, prob), noise in zip(actions, dirichlet_noise):
            noise_prob = epsilon * noise + (1 - epsilon) * prob
            noise_action_probs.append((action, noise_prob))
        return noise_action_probs

    def mcts_search(self, game_env: BaseEnv):
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
        root: SearchNode = SearchNode(None, game_env.current_player(), 1)
        for _ in range(self.max_simulations):
            visit_path, working_state_env = self._apply_tree_policy(
                root, game_env)
            if working_state_env.is_terminal():
                returns = working_state_env.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state_env)
                solved = False

            while visit_path:
                # For chance nodes, walk up the tree to find the decision-maker.
                decision_node_idx = -1
                # Chance node targets are for the respective decision-maker.
                target_return = returns[visit_path[decision_node_idx].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    # If any have max utility (won?), or all children are solved,
                    # choose the one best for the player choosing.
                    best = None
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
