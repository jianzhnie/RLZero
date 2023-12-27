from __future__ import print_function

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append('..')
from games.gomoku import Game, GomokuEnv
from games.gomoku.alphazero_agent import AlphaZeroAgent
from rlzero.mcts.alphazero_mcts import AlphaZeroPlayer
from rlzero.mcts.human_player import HumanPlayer
from rlzero.mcts.mcts_deepmind import MCTSBot, RandomRolloutEvaluator
from rlzero.mcts.player import Player
from rlzero.mcts.rollout_mcts import RolloutPlayer


class MCTSPlayer(Player):

    def __init__(self,
                 max_simulations=1000,
                 player_id: int = 0,
                 player_name: str = '') -> None:
        super().__init__(player_id, player_name)

        game_env = GomokuEnv(board_size=5, n_in_row=4, start_player_idx=0)
        evaluator = RandomRolloutEvaluator(n_rollouts=1)
        self.mcts_bot: MCTSBot = MCTSBot(
            game_env,
            uct_c=2,
            max_simulations=max_simulations,
            evaluator=evaluator,
            child_selection_method='puct',
            solve=False,
            verbose=False,
        )

    def set_player_id(self, player_id):
        self.player_id = player_id

    def get_player_id(self):
        return self.player_id

    def get_player_name(self):
        return self.player_name

    def get_action(self, game_env, **kwargs):
        sensible_moves = game_env.leagel_actions()
        if len(sensible_moves) > 0:
            action = self.mcts_bot.step(game_env)
            return action
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'DeepMindMCTS, id: {}, name: {}.'.format(
            self.get_player_id(), self.get_player_name())


def mcts_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(board_size=4, n_in_row=3, start_player_idx=0)
    game = Game(board)
    # 加载模型
    mcts_player1 = MCTSPlayer(max_simulations=1000, player_name='MCTS_0')
    # 两个AI对打
    mcts_player2 = MCTSPlayer(max_simulations=100, player_name='MCTS_1')
    # 开始对打
    game.start_play(mcts_player1, mcts_player2, start_player=0)


def human_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(board_size=3, n_in_row=3, start_player_idx=0)
    game = Game(board)
    mcts_player1 = HumanPlayer(player_name='Human')
    mcts_player2 = RolloutPlayer(n_playout=1000, player_name='MCTS')
    # 开始对打
    game.start_play(mcts_player1, mcts_player2, start_player=0)


def alphazero_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(3, 3, 3)
    game = Game(board)
    # 加载模型
    alphazero_agent = AlphaZeroAgent(board_size=3)
    alphazero_plyer = AlphaZeroPlayer(alphazero_agent.policy_value_fn,
                                      n_playout=1,
                                      player_name='AlphaZero')
    # 两个AI对打
    mcts_player2 = RolloutPlayer(n_playout=10, player_name='MCTS_2')
    # 开始对打
    game.start_play(alphazero_plyer, mcts_player2, start_player=0)


if __name__ == '__main__':
    # alphazero_vs_mcts()
    mcts_vs_mcts()
    # human_vs_mcts()
