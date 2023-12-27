from __future__ import print_function

import numpy as np

from games.gomoku.gomoku_env import GomokuEnv
from rlzero.mcts.player import Player

from .visual_tool import VisualTool


class Game(object):
    """game server."""

    def __init__(
        self,
        game_env: GomokuEnv,
        is_visualize: bool = False,
    ) -> None:
        self.game_env = game_env
        self.visualTool = None
        if is_visualize:
            self.visualTool = VisualTool(board_size=[
                self.game_env.board_size, self.game_env.board_size
            ])

    def set_player_symbol(self, start_player) -> None:
        """show board, set player symbol X OR O."""
        p1, p2 = self.game_env.players
        if self.game_env.players[start_player] == p1:
            self.player1_symbol = 'X'
            self.player2_symbol = 'O'
        else:
            self.player1_symbol = 'O'
            self.player2_symbol = 'X'

    def show(self) -> None:
        self.visualTool.draw()

    def graphic_visualTool(
        self,
        game_env: GomokuEnv,
        player1: Player,
        player2: Player,
    ):
        """Draw the board and show game info."""
        loc = self.game_env.move_to_location(self.game_env.last_move)
        self.visualTool.graphic(loc[0], loc[1])

    def graphic(
        self,
        game_env: GomokuEnv,
        player1: Player,
        player2: Player,
    ):
        """Draw the board and show game info."""
        board_size = game_env.board_size
        player1_id = player1 if isinstance(player1,
                                           int) else player1.get_player_id()
        player2_id = player2 if isinstance(player2,
                                           int) else player2.get_player_id()

        print('Player', player1, self.player1_symbol.rjust(3))
        print('Player', player2, self.player2_symbol.rjust(3))
        print()
        for x in range(board_size):
            print('{0:8}'.format(x), end='')
        print('\r\n')
        for i in range(board_size - 1, -1, -1):
            print('{0:4d}'.format(i), end='')
            for j in range(board_size):
                loc = i * board_size + j
                p = game_env.states.get(loc, -1)
                if p == player1_id:
                    print(self.player1_symbol.center(8), end='')
                elif p == player2_id:
                    print(self.player2_symbol.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(
        self,
        player1: Player,
        player2: Player,
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
        self.set_player_symbol(start_player)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.game_env, player1, player2)
        while True:
            player_in_turn = players[self.game_env.current_player()]
            move = player_in_turn.get_action(self.game_env)
            self.game_env.step(move)
            if is_shown:
                self.graphic(self.game_env, player1, player2)
            end, winner = self.game_env.game_end_winner()
            if end:
                if is_shown:
                    if winner != -1:
                        print('Game end. Winner is', players[winner])
                    else:
                        print('Game end. Tie')
                return winner

    def start_self_play(
        self,
        player: Player,
        is_shown: bool = False,
        temperature: float = 1e-3,
    ):
        """start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training."""
        self.game_env.reset()
        p1, p2 = self.game_env.players
        states, mcts_probs, current_players = [], [], []
        self.set_player_symbol(start_player=0)
        while True:
            move, move_probs = player.get_action(self.game_env,
                                                 temperature=temperature,
                                                 return_prob=True)
            # store the data
            states.append(self.game_env.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.game_env.current_player())
            # perform a move
            self.game_env.step(move)
            if is_shown:
                self.graphic(self.game_env, p1, p2)
            end, winner = self.game_env.game_end_winner()
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

    def __str__(self):
        return 'Game'
