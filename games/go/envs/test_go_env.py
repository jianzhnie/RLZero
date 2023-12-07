import pytest
from go_env import GoEnv
from pettingzoo.classic import go_v5


@pytest.mark.envtest
class TestGoEnv:

    def test_naive(self):
        env = GoEnv(board_size=9, komi=7.5, render_mode='human')
        print(
            'NOTE:actions are counted by column, such as action 9, which is the second column and the first row'
        )
        env.reset()
        for i in range(1000):
            """player 1."""
            # action = env.human_to_action()
            action = env.random_action()
            print('player 1 (black_0): ', action)
            obs, reward, done, info = env.step(action)
            assert isinstance(obs, dict)
            assert isinstance(done, bool)
            assert isinstance(reward, float)
            # env.render()
            if done:
                if reward > 0:
                    print('player 1 (black_0) win')
                else:
                    print('draw')
                break
            """player 2"""
            action = env.random_action()
            print('player 2 (white_0): ', action)
            obs, reward, done, info = env.step(action)
            # env.render()
            if done:
                if reward > 0:
                    print('player 2 (white_0) win')
                else:
                    print('draw')
                break


def test_pettingzoo():
    env = go_v5.env(render_mode='human')
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation['action_mask']
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        env.step(action)
    env.close()


def test_goenv():
    env = GoEnv(board_size=9, render_mode='human')
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation['action_mask']
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        env.step(action)
    env.close()


if __name__ == '__main__':
    env = TestGoEnv()
    env.test_naive()
    # test_pettingzoo()
    # test_goenv()
