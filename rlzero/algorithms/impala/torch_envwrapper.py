"""The environment class for MonoBeast."""

from typing import Dict

import gymnasium as gym
import numpy as np
import torch


def _format_obs(obs: np.ndarray) -> torch.Tensor:
    """Convert numpy observation to a formatted torch tensor."""
    obs = torch.from_numpy(obs)
    return obs.view((1, 1) + obs.shape)  # Reshape to [1, 1, ...]


class TorchEnvWrapper(object):
    """A wrapper class for interacting with a Gym environment.

    Attributes:
        gym_env (gym.Env): The Gym environment instance.
        episode_return (torch.Tensor): Total reward for the current episode.
        episode_step (torch.Tensor): Step count for the current episode.
    """

    def __init__(self, gym_env: gym.Env) -> None:
        """Initialize the Environment with a Gym environment."""
        self.gym_env = gym_env
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment and return the initial observation.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the initial observation, action, reward,
            done flag, episode return, and step count.
        """
        obs, _ = self.gym_env.reset()
        reward = torch.zeros(1, 1)
        action = torch.zeros(1, 1, dtype=torch.int64)
        done = torch.ones(1, 1, dtype=torch.uint8)
        obs = _format_obs(obs)
        return dict(
            obs=obs,
            reward=reward,
            action=action,
            done=done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
        )

    def step(self, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Take a step in the environment using the provided action.

        Args:
            action (torch.Tensor): The action to take.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the new observation, action, reward,
            done flag, episode return, and step count.
        """
        obs, reward, terminated, truncated, _ = self.gym_env.step(
            action.item())
        self.episode_step += 1
        self.episode_return += reward

        done = terminated or truncated
        if done:
            obs, _ = self.gym_env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = _format_obs(obs)
        reward_tensor = torch.tensor(reward).view(1, 1)
        done_tensor = torch.tensor(done).view(1, 1)

        return dict(
            obs=obs,
            reward=reward_tensor,
            action=action,
            done=done_tensor,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
        )

    def close(self) -> None:
        """Close the Gym environment."""
        self.gym_env.close()
