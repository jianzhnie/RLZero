from collections import deque, namedtuple
from typing import Any, Dict, Tuple

import numpy as np

# Define a namedtuple for transitions
Transition = namedtuple('Transition',
                        ('obs', 'action', 'reward', 'next_obs', 'done'))


class PrioritizedReplayBuffer:
    """A class to represent a Prioritized Replay Buffer for reinforcement
    learning.

    This buffer stores experiences (transitions) and allows sampling based on
    transition priorities to improve learning efficiency.

    Attributes:
        buffer_size (int): Maximum size of the buffer.
        alpha (float): Controls how much prioritization is used.
        beta (float): Controls how much importance sampling is used.
        buffer (deque): Stores transitions (state, action, reward, next state, done).
        prior_buffer (deque): Stores the priorities associated with each transition.
    """

    def __init__(self,
                 buffer_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4) -> None:
        """Initializes the PrioritizedReplayBuffer with a specified size and
        prioritization settings.

        Args:
            buffer_size (int): The maximum number of transitions the buffer can store.
            alpha (float): Prioritization exponent. Higher values prioritize more important samples.
            beta (float): Importance-sampling exponent. Used to reduce bias when sampling.
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
    ) -> None:
        """Adds a new transition along with its priority to the buffer.

        Args:
            obs (Any): Current observation/state.
            action (Any): Action taken in the current state.
            reward (float): Reward received after taking the action.
            next_obs (Any): Next observation/state after the action.
            done (bool): Flag indicating whether the episode has ended.
            prior (float): The priority assigned to this transition.
        """
        transition = Transition(obs, action, reward, next_obs, done)
        self.buffer.append(transition)
        self.prior_buffer.append(prior)

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """Samples a batch of transitions based on their priorities.

        The higher the priority, the more likely a transition is to be sampled.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple containing:
            - indices (np.ndarray): Indices of the sampled transitions.
            - batch (Dict[str, np.ndarray]): Dictionary of the sampled transitions (states, actions, etc.).
            - weights (np.ndarray): Importance sampling weights for each sampled transition.
        """
        if len(self.buffer) == 0:
            raise ValueError('The buffer is empty. Cannot sample.')

        # Convert deque of priorities to numpy array for manipulation
        prior_probs = np.array(self.prior_buffer)

        # Compute sampling probabilities with prioritization
        scaled_priorities = prior_probs**self.alpha
        sampling_probs = scaled_priorities / np.sum(scaled_priorities)

        # Randomly sample indices based on the computed probabilities
        indices = np.random.choice(len(self.buffer),
                                   size=batch_size,
                                   p=sampling_probs)

        # Sample transitions and compute importance sampling weights
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * sampling_probs[indices])**(-self.beta)
        weights /= weights.max()  # Normalize weights

        # Extract individual components from sampled transitions
        obs, actions, rewards, next_obs, dones = zip(*samples)
        batch = {
            'states': np.array(obs),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_obs),
            'dones': np.array(dones),
        }

        return np.array(indices), batch, weights

    def update_priorities(self, indices: np.ndarray,
                          priorities: np.ndarray) -> None:
        """Updates the priorities of specific transitions in the buffer.

        Args:
            indices (np.ndarray): The indices of the transitions to update.
            priorities (np.ndarray): The new priorities for each transition.
        """
        if len(indices) != len(priorities):
            raise ValueError(
                'Indices and priorities must have the same length.')

        for idx, priority in zip(indices, priorities):
            if idx < 0 or idx >= len(self.prior_buffer):
                raise IndexError(
                    f'Index {idx} is out of bounds for the priority buffer.')
            self.prior_buffer[idx] = priority

    def __len__(self) -> int:
        """Returns the current number of transitions in the buffer.

        Returns:
            int: The number of transitions stored in the buffer.
        """
        return len(self.buffer)
