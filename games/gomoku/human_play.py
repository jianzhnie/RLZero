"""human VS AI models Input your move in the format: 2,3.

@author: Junxiao Song
"""

from __future__ import print_function

import sys

from game import GomokuGame

sys.path.append('../../')
from muzero.mcts.mcts_pure import MCTSPlayer as MCTS_Pure


class Human(object):
    """human player."""
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input('Your move: ')
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(',')]
            move = board.location_to_move(location)
        except Exception as e:
            print(e)
            move = -1
        if move == -1 or move not in board.availables:
            print('invalid move')
            move = self.get_action(board)
        return move

    def __str__(self):
        return 'Human {}'.format(self.player)


def run():
    n = 5
    width, height = 8, 8
    try:
        gomokugame = GomokuGame(width=width, height=height, n_in_row=n)
        #  human VS AI
        mcts_player = MCTS_Pure(c_puct=5, n_playout=100)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        gomokugame.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
