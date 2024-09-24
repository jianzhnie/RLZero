"""The environment class for MonoBeast."""

import torch
import numpy as np
import gymnasium as gym


def _format_obs(obs: np.ndarray) -> torch.Tensor:
    obs = torch.from_numpy(obs)
    return obs.view((1, 1) + obs.shape)


class Environment:
    def __init__(self, gym_env: gym.Env) -> None:
        self.gym_env = gym_env
        self.episode_return = None
        self.episode_step = None

    def reset(self) -> dict[str, torch.Tensor]:
        obs, _ = self.gym_env.reset()
        reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        action = torch.zeros(1, 1, dtype=torch.int64)
        done = torch.ones(1, 1, dtype=torch.uint8)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = _format_obs(obs)
        return dict(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
        )

    def step(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        obs, reward, terminated, truncated, info = self.gym_env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        done = terminated or truncated
        if done:
            obs, _ = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = _format_obs(obs)
        reward = torch.tensor(reward).view(1, 1)
        done = torch.tensor(done).view(1, 1)

        return dict(
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            episode_return=episode_return,
            episode_step=episode_step,
        )

    def close(self) -> None:
        self.gym_env.close()
