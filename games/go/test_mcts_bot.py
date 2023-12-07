import sys

from envs.go_env import GoEnv

sys.path.append('../../')
from rlzero.mcts.mcts_bot import MCTSBot


def test_go_mctsbot_vs_mctsbot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # Initialize the game, where there are two players: player 1 and player 2.
    env = GoEnv(board_size=5, komi=0)
    # Reset the environment, set the board to a clean board and the  start player to be player 1.
    env.reset(seed=42)
    mcts_player_0 = MCTSBot(env, 'mcts_bot0', num_simulations)
    mcts_player_1 = MCTSBot(env, 'mcts_bot1', num_simulations)
    # player_index = 0, player = 1
    # Set player 1 to move first.
    player_index = 0
    for player_index in env.agent_iter():
        print('player_index: ', player_index)
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            if player_index == 0:
                action = mcts_player_0.get_actions()
            else:
                action = mcts_player_1.get_actions()
        env.step(action)
    env.close()


if __name__ == '__main__':
    test_go_mctsbot_vs_mctsbot()
