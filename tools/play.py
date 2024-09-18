from __future__ import print_function

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append('..')
from rlzero.games.gomoku import GameControl, GomokuEnv
from rlzero.games.gomoku.alphazero_agent import AlphaZeroAgent
from rlzero.mcts.alphazero_mcts import AlphaZeroPlayer
from rlzero.mcts.deepmind_mcts import MCTSBot
from rlzero.mcts.player import HumanPlayer
from rlzero.mcts.rollout_mcts import RolloutPlayer


def mcts_vs_mcts():
    # 初始化棋盘
    game_env = GomokuEnv(board_size=4, n_in_row=3, start_player_idx=0)
    game_control = GameControl(game_env)
    # 加载模型
    mcts_player1 = MCTSBot(game_env=game_env,
                           max_simulations=100,
                           player_name='MCTSBot_0')
    # 两个AI对打
    mcts_player2 = MCTSBot(game_env=game_env,
                           max_simulations=10,
                           player_name='MCTSBot_1')
    # 开始对打
    game_control.start_play(mcts_player1, mcts_player2, start_player=0)


def human_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(board_size=3, n_in_row=3, start_player_idx=0)
    game_control = GameControl(board)
    mcts_player1 = HumanPlayer(player_name='Human')
    mcts_player2 = RolloutPlayer(n_playout=1000, player_name='MCTS')
    # 开始对打
    game_control.start_play(mcts_player1, mcts_player2, start_player=0)


def alphazero_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(3, 3, 3)
    game_control = GameControl(board)
    # 加载模型
    alphazero_agent = AlphaZeroAgent(board_size=3)
    alphazero_plyer = AlphaZeroPlayer(alphazero_agent.policy_value_fn,
                                      n_playout=1,
                                      player_name='AlphaZero')
    # 两个AI对打
    mcts_player2 = RolloutPlayer(n_playout=10, player_name='MCTS_2')
    # 开始对打
    game_control.start_play(alphazero_plyer, mcts_player2, start_player=0)


if __name__ == '__main__':
    # alphazero_vs_mcts()
    mcts_vs_mcts()
    # human_vs_mcts()
