from __future__ import print_function

import sys

sys.path.append('..')
from games.game import Game
from games.gomoku.alphazero_agent import AlphaZeroAgent
from games.gomoku.gomoku_env import GomokuEnv
from rlzero.mcts.alphazero_mcts import AlphaZeroPlayer
from rlzero.mcts.rollout_mcts import RolloutPlayer


def main():
    # 初始化棋盘
    board = GomokuEnv()
    game = Game(board)
    # 加载模型
    alphazero_agent = AlphaZeroAgent(board.width, board.height)
    alphazero_plyer = AlphaZeroPlayer(alphazero_agent.policy_value_fn)
    # 两个AI对打
    mcts_player = RolloutPlayer()
    # 开始对打
    game.start_play(mcts_player, alphazero_plyer, start_player=0)


if __name__ == '__main__':
    main()
