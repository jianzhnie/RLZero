from __future__ import print_function

import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append('..')
from games.game import Game
from games.gomoku.alphazero_agent import AlphaZeroAgent
from games.gomoku.gomoku_env import GomokuEnv
from rlzero.mcts.alphazero_mcts import AlphaZeroPlayer
from rlzero.mcts.human_player import HumanPlayer
from rlzero.mcts.rollout_mcts import RolloutPlayer


def mcts_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(3, 3, 3)
    game = Game(board)
    # 加载模型
    # alphazero_agent = AlphaZeroAgent(board.width, board.height)
    # alphazero_plyer = AlphaZeroPlayer(alphazero_agent.policy_value_fn)
    mcts_player1 = RolloutPlayer(n_playout=2000, player_name='MCTS_1')
    # 两个AI对打
    mcts_player2 = RolloutPlayer(n_playout=1000, player_name='MCTS_2')
    # 开始对打
    game.start_play(mcts_player1, mcts_player2, start_player=0)


def human_vs_mcts():
    # 初始化棋盘
    board = GomokuEnv(3, 3, 3)
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
    alphazero_agent = AlphaZeroAgent(board.width, board.height)
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
    human_vs_mcts()
