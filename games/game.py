from __future__ import print_function

import numpy as np

from games.gomoku.gomoku_env import GomokuEnv


class Game(object):
    """game server."""

    def __init__(self, game_env: GomokuEnv):
        self.game_env = game_env

    def graphic(self, game_env: GomokuEnv, player1: int, player2: int):
        """Draw the board and show game info."""
        width = game_env.width
        height = game_env.height

        print('Player', player1, 'with X'.rjust(3))
        print('Player', player2, 'with O'.rjust(3))
        print()
        for x in range(width):
            print('{0:4}'.format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print('{0:4d}'.format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = game_env.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(
        self,
        player1: int,
        player2: int,
        start_player: int = 0,
        is_shown: bool = True,
    ) -> int:
        """start a game between two players."""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.game_env.reset()
        p1, p2 = self.game_env.players
        player1.set_player_id(p1)
        player2.set_player_id(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.game_env, player1.player, player2.player)
        while True:
            current_player = self.game_env.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.game_env)
            self.game_env.step(move)
            if is_shown:
                self.graphic(self.game_env, player1.player, player2.player)
            end, winner = self.game_env.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print('Game end. Winner is', players[winner])
                    else:
                        print('Game end. Tie')
                return winner

    def start_self_play(self, player, is_shown=0, temperature=1e-3):
        """start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training."""
        self.game_env.reset()
        p1, p2 = self.game_env.players
        states, mcts_probs, current_players = [], [], []
        while True:
            move, move_probs = player.get_action(self.game_env,
                                                 temperature=temperature,
                                                 return_prob=True)
            # store the data
            states.append(self.game_env.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.game_env.current_player)
            # perform a move
            self.game_env.step(move)
            if is_shown:
                self.graphic(self.game_env, p1, p2)
            end, winner = self.game_env.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print('Game end. Winner is player:', winner)
                    else:
                        print('Game end. Tie')
                return winner, zip(states, mcts_probs, winners_z)
