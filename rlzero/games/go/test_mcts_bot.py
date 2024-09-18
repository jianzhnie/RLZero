import sys

from go_env import GoEnv

sys.path.append('../../')
from rlzero.mcts.deepmind_mcts import DeepMindMCTS


def test_go_mctsbot_vs_mctsbot(num_simulations=50):
    """
    Overview:
        A tictactoe game between mcts_bot and rule_bot, where rule_bot take the first move.
    Arguments:
        - num_simulations (:obj:`int`): The number of the simulations required to find the best move.
    """
    # Initialize the game, where there are two players: player 1 and player 2.
    env = GoEnv(board_size=9, komi=7.5, render_mode='human')
    # Reset the environment, set the board to a clean board and the  start player to be player 1.
    env.reset(seed=42)
    print(env.legal_actions())
    mcts_player1 = DeepMindMCTS(env,
                                max_simulations=num_simulations,
                                verbose=True)
    print('after init mcts_player1')
    print(mcts_player1.game_env.legal_actions())
    for (agent_index, agent) in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            if agent_index == 0:
                print('get plyer1 action: ')
                action = mcts_player1.step(env)
            else:
                mask = observation['action_mask']
                action = env.action_space(agent).sample(mask)
        print(f"Agent {agent}'s turn. Action: {action}")
        env.step(action)
    env.close()


if __name__ == '__main__':
    test_go_mctsbot_vs_mctsbot()
