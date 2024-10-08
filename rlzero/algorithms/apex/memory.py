from collections import deque, namedtuple
from typing import Any, Dict, Tuple

import numpy as np

# Define a namedtuple for transitions
Transition = namedtuple('Transition',
                        ('obs', 'action', 'reward', 'next_obs', 'done'))


class PrioritizedReplayBuffer:
    """A class to represent a Prioritized Replay Buffer for reinforcement
    learning.

    Attributes:
        buffer_size (int): Maximum size of the buffer.
        alpha (float): Exponent to control the priority distribution.
        beta (float): Exponent to control the importance sampling weights.
        buffer (deque): A deque to store transitions.
        prior_buffer (deque): A deque to store priorities.
    """

    def __init__(self,
                 buffer_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        """Initializes the PrioritizedReplayBuffer with given parameters.

        Args:
            buffer_size (int): Maximum size of the buffer.
            alpha (float): Exponent to control the priority distribution.
            beta (float): Exponent to control the importance sampling weights.
        """
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=buffer_size)
        self.prior_buffer = deque(maxlen=buffer_size)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: float,
        next_obs: Any,
        done: bool,
        prior: float,
    ):
        """Adds a new transition to the buffer along with its priority.

        Args:
            obs (Any): Current observation.
            action (Any): Action taken.
            reward (float): Reward received.
            next_obs (Any): Next observation.
            done (bool): Whether the episode is done.
            prior (float): Priority of the transition.
        """
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
        self.prior_buffer.append(prior)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Samples a batch of transitions from the buffer based on priorities.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
                - indices: Indices of sampled transitions.
                - batch: Dictionary containing states, actions, rewards, next_states, and dones.
                - weights: Importance sampling weights.
        """
        prior_prob = np.array(self.prior_buffer)
        prior_prob = prior_prob**self.alpha
        prior_prob = prior_prob / np.sum(prior_prob)
        indices = np.random.choice(len(self.buffer),
                                   size=batch_size,
                                   p=prior_prob)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * prior_prob[indices])**(-self.beta)
        weights /= weights.max()

        obs, actions, rewards, next_obs, dones = zip(*samples)
        batch = dict(
            states=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_obs),
            dones=np.array(dones),
        )
        return indices, batch, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates the priorities of the specified indices.

        Args:
            indices (np.ndarray): Indices of transitions to update.
            priorities (np.ndarray): New priorities for the specified indices.
        """
        for idx, priority in zip(indices, priorities):
            self.prior_buffer[idx] = priority

    def __len__(self) -> int:
        """Returns the current size of the buffer.

        Returns:
            int: Number of transitions in the buffer.
        """
        return len(self.buffer)
