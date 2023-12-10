from .player import Player
from .rollout_mcts import RolloutMCTS


class RolloutPlayer(Player):

    def __init__(self, nplays=1000, c_puct=5, player_id=0, player_name=''):
        Player.__init__(self, player_id, player_name)
        self.mcts = RolloutMCTS(nplays, c_puct)

    def reset_player(self):
        self.mcts.reuse(-1)

    def play(self, board, **kwargs):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.simulate(board)
            self.mcts.reuse(-1)
            return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'RolloutPlayer {}{}'.format(self.get_player_id(),
                                           self.get_player_name())
