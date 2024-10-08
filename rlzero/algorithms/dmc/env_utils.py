"""This file provides a wrapper around the original environment to make it
easier to use.

When a game finishes, the environment is automatically reset instead of needing
manual reset. The observations are also formatted and moved to the appropriate
device (CPU/GPU).
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch


@dataclass
class StepResult:
    player_id: str
    obs: Dict[str, Any]
    done: torch.Tensor
    episode_return: torch.Tensor
    obs_x_no_action: torch.Tensor
    obs_z: torch.Tensor


def format_observation(
    observation: Dict[str, Any], device: Union[str, torch.device]
) -> Tuple[str, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Utility function to process observations and move them to the specified
    device (CPU or CUDA).

    Args:
        obs (Dict[str, Any]): The observation from the environment, containing 'player', 'x_batch', 'z_batch', etc.
        device (Union[str, torch.device]): The device to move the data to, either 'cpu' or 'cuda'.

    Returns:
        Tuple[str, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]: Processed observation with tensors moved to the specified device.
    """
    player_id = observation['position']

    # Ensure the device is correctly formatted
    if device != 'cpu':
        device = 'cuda:' + str(device)
    device = torch.device(device)

    # Move observations to the specified device
    x_batch = torch.from_numpy(observation['x_batch']).to(device)
    z_batch = torch.from_numpy(observation['z_batch']).to(device)
    x_no_action = torch.from_numpy(
        observation['x_no_action'])  # Not moved to device
    z = torch.from_numpy(observation['z'])  # Not moved to device

    # Return formatted observation
    formatted_obs = {
        'x_batch': x_batch,
        'z_batch': z_batch,
        'legal_actions': observation['legal_actions'],
    }
    return player_id, formatted_obs, x_no_action, z


class EnvWrapper:
    """A wrapper for the game environment that automatically resets when the
    game finishes.

    It also handles formatting and moving data to the appropriate device (CPU
    or GPU).
    """

    def __init__(self, env: Any, device: Union[str, torch.device]):
        """Initialize the Environment wrapper.

        Args:
            env (Any): The original environment instance.
            device (Union[str, torch.device]): The device to move data to, either 'cpu' or 'cuda'.
        """
        self.env = env
        self.device = device
        self.episode_return = None  # Tracks the cumulative reward for an episode

    @torch.no_grad()
    def initial(
            self
    ) -> Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Resets the environment and returns the initial observation and
        state.

        Returns:
            Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Initial player, observation, and additional state information.
        """
        # Reset the environment and format the initial observation
        initial_player_id, initial_obs, x_no_action, z = format_observation(
            self.env.reset(), self.device)

        # Initialize episode return and done flag
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)

        return (
            initial_player_id,
            initial_obs,
            {
                'done': initial_done,
                'episode_return': self.episode_return,
                'obs_x_no_action': x_no_action,
                'obs_z': z,
            },
        )

    @torch.no_grad()
    def step(
        self, action: torch.Tensor
    ) -> Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Takes a step in the environment using the provided action, processes
        the result, and resets the environment if the episode is done.

        Args:
            action (torch.Tensor): The action to be taken in the environment.

        Returns:
            Tuple[str, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: The new player, observation, and additional state information.
        """
        # Take a step in the environment
        original_obs, reward, done, _ = self.env.step(action)

        # Update the cumulative reward
        self.episode_return += reward
        episode_return = self.episode_return

        # Reset environment if the episode is finished
        if done:
            original_obs = self.env.reset()  # Reset environment
            self.episode_return = torch.zeros(1, 1)  # Reset episode return

        # Format the new observation
        player_id, obs, x_no_action, z = format_observation(
            original_obs, self.device)

        # Convert reward and done to tensors
        done_tensor = torch.tensor(done).view(1, 1)

        return (
            player_id,
            obs,
            {
                'done': done_tensor,
                'episode_return': episode_return,
                'obs_x_no_action': x_no_action,
                'obs_z': z,
            },
        )

    def close(self) -> None:
        """Closes the environment."""
        self.env.close()
