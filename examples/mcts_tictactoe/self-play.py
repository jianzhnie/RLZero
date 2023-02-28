import random

from .mcts_chatgpt import Node


class MCTSAgent:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def get_move(self, game):
        root = Node(game)
        for i in range(self.num_simulations):
            node = root
            state = game.copy()
            # Selection
            while node.untried_moves == [] and node.children != []:
                node = node.select_child()
                move = node.move
                state.make_move(move)
            # Expansion
            if node.untried_moves != []:
                move = random.choice(node.untried_moves)
                state.make_move(move)
                node = node.add_child(move, state)
            # Simulation
            while not state.is_game_over():
                move = random.choice(state.get_legal_moves())
                state.make_move(move)
            # Backpropagation
            while node is not None:
                node.update(state.get_result(node.player))
                node = node.parent
        # Choose best move
        return max(root.children, key=lambda n: n.visits).move


class SelfPlay:
    def __init__(self, game, player1, player2, num_games=100):
        self.game = game
        self.player1 = player1
        self.player2 = player2
        self.num_games = num_games

    def play_game(self):
        state = self.game()
        players = [self.player1, self.player2]
        random.shuffle(players)
        while not state.is_game_over():
            move = players[state.current_player].get_move(state)
            state.make_move(move)
        return state.get_result()

    def play(self):
        results = []
        for i in range(self.num_games):
            result = self.play_game()
            results.append(result)
        return results
