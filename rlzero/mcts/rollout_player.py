from .player import Player
from .rollout_mcts import RolloutMCTS


class RolloutPlayer(Player):

    def __init__(self, n_playout=1000, c_puct=5, player_id=0, player_name=''):
        Player.__init__(self, player_id, player_name)
        self.mcts = RolloutMCTS(n_playout, c_puct)

    def reset_player(self) -> None:
        self.mcts.update_with_move(-1)

    def play(self, game_env, **kwargs):
        sensible_moves = game_env.availables
        if len(sensible_moves) > 0:
            move = self.mcts.simulate(game_env)
            self.mcts.update_with_move(move)
            return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'RolloutPlayer {}{}'.format(self.get_player_id(),
                                           self.get_player_name())
