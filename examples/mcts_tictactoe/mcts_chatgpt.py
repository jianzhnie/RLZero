import math
import random
from typing import Dict


class Node:

    def __init__(self, parent=None, action=None, prior_p=None):
        self.parent = parent
        self.action = action
        self.prior_p = prior_p
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

    def select_child(self):
        """Select a child node based on the UCB1 formula."""
        if not self.children:
            raise ValueError('Node has no children.')
        return max(self.children, key=lambda child: child.ucb_score())

    def expand(self, action_priors: Dict[str, float]):
        """Expand the node by adding children for each possible action."""
        for action, prob in action_priors.items():
            self.children.append(Node(self, action, prob))

    def update(self, reward: float):
        """Update the node's total reward and visit count."""
        self.visits += 1
        self.total_reward += reward

    def ucb_score(self, exploration_value: float = 1.4):
        """Compute the node's UCB1 score."""
        if self.visits == 0:
            return float('inf')
        else:
            exploitation_score = self.total_reward / self.visits
            exploration_score = math.sqrt(
                math.log(self.parent.visits) / self.visits)
        return exploitation_score + exploration_score * exploration_value

    @property
    def fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def __repr__(self):
        return 'MCTSNode'


class MonteCarloTreeSearch:

    def __init__(self, state, exploration_weight=1.4):
        self.root = Node(state)
        self.exploration_weight = exploration_weight

    def select(self):
        """Select a leaf node in the tree to expand."""
        current = self.root
        while not current.is_leaf():
            current = current.select_child()
        return current

    def expand(self, node: Node):
        """Expand the selected node by adding all possible children."""
        if node.state.is_terminal():
            return
        for move in node.untried_moves:
            node.add_child(move)

    def simulate(self, node: Node):
        """Simulate a random game from the current state of the node."""
        current_state = node.state
        while not current_state.is_terminal(current_state):
            action = random.choice(current_state.get_legal_moves())
            current_state = current_state.make_move(action)
        return current_state.get_winner()

    def backpropagate(self, node: Node, reward: float):
        """Backpropagate the result of a simulation to all parent nodes."""
        while node is not None:
            node.update(reward)
            node = node.parent
            reward = -reward

    def get_best_child(self):
        """Get the child node with the highest expected reward."""
        best_score = float('-inf')
        best_children = []
        for child in self.root.children:
            score = child.reward / child.visits
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        return random.choice(best_children)

    def get_best_move(self):
        best_score = float('-inf')
        best_moves = []
        for child in self.root.children:
            if child.visits > best_score:
                best_score = child.visits
                best_moves = [child.move]
            elif child.visits == best_score:
                best_moves.append(child.move)
        return random.choice(best_moves)

    def search(self, num_simulations):
        for i in range(num_simulations):
            # Selection phase
            node = self.select(self.root)
            # Expansion phase
            self.expand(node)
            # Simulation phase
            reward = self.simulate(node)
            # Backpropagation phase
            self.backpropagate(node, reward)


class TicTacToe:

    def __init__(self):
        self.board = [' '] * 9
        self.winning_combinations = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],  # rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],  # columns
            [0, 4, 8],
            [2, 4, 6]  # diagonals
        ]
        self.current_player = 'X'

    def get_legal_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def is_valid_move(self, move):
        return self.board[move] == ' '

    def make_move(self, move):
        if not self.is_valid_move:
            raise ValueError('Invalid move')
        self.board[move] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def get_winner(self):
        if self.is_winner('X'):
            return 'X'
        elif self.is_winner('O'):
            return 'O'
        elif self.is_tie():
            return None

    def get_reward(self):
        if self.is_winner('X'):
            return 1
        elif self.is_winner('O'):
            return -1
        else:
            return 0

    def is_winner(self, player):
        for combination in self.winning_combinations:
            if all(self.board[i] == player for i in combination):
                return True
        return False

    def is_tie(self):
        return ' ' not in self.board

    def is_terminal(self):
        return not self.get_legal_moves() or self.is_winner(
            'X') or self.is_winner('O')

    def display_board(self):
        """Displays the current state of the game board."""
        print(' %s | %s | %s ' % (self.board[0], self.board[1], self.board[2]))
        print('-----------')
        print(' %s | %s | %s ' % (self.board[3], self.board[4], self.board[5]))
        print('-----------')
        print(' %s | %s | %s ' % (self.board[6], self.board[7], self.board[8]))


def play_tictatoe(human_play=False):

    game = TicTacToe()
    game.display_board()

    while not game.is_terminal():
        if human_play:
            move = int(input('Enter position (0-8): '))
            if game.is_valid_move(move):
                game.make_move(move)
            else:
                print('Position not available.')
        else:
            available_actions = game.get_legal_moves()
            move = random.choice(available_actions)
            game.make_move(move)
        print('==' * 10)
        game.display_board()

    winner = game.get_winner()
    if winner:
        print('Player %s wins!' % winner)
    else:
        print('Tie game!')


if __name__ == '__main__':
    play_tictatoe(human_play=False)
