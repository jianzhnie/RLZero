import traceback
from collections import deque
from typing import Dict, Iterator, List, Union

import gymnasium as gym
import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn

from rlzero.agents.dmc.env_utils import EnvWrapper
from rlzero.envs.doudizhu.env import _cards2array
from rlzero.models.doudizhu import DouDiZhuModel
from rlzero.utils.logger_utils import get_logger

logger = get_logger('rlzero.agents.dmc.utils')

# Dictionary mapping card values to column indices
Card2Column = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    17: 12,
}

# Dictionary mapping the number of ones to a specific binary array
NumOnes2Array = {
    0: np.array([0, 0, 0, 0]),
    1: np.array([1, 0, 0, 0]),
    2: np.array([1, 1, 0, 0]),
    3: np.array([1, 1, 1, 0]),
    4: np.array([1, 1, 1, 1]),
}


def create_optimizers(
    learning_rate: float,
    momentum: float,
    epsilon: float,
    alpha: float,
    learner_model: Dict[str, nn.Module],
) -> Dict[str, torch.optim.Optimizer]:
    """Create optimizers for three different positions: 'landlord',
    'landlord_up', and 'landlord_down'.

    Args:
        learning_rate: The learning rate for the optimizer.
        momentum: The momentum for the optimizer.
        epsilon: The epsilon for the optimizer.
        alpha: The alpha for the optimizer.
        learner_model: The model with separate parameter groups for each position.

    Returns:
        dict: A dictionary with optimizers for each position.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']
    optimizers = {}

    for position in positions:
        # Retrieve the parameters for the specific position from the learner model
        position_parameters = getattr(learner_model, position).parameters()
        # Create an RMSprop optimizer for the current position
        optimizer = torch.optim.RMSprop(
            position_parameters,
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha,
        )

        # Store the optimizer in the dictionary with the position as the key
        optimizers[position] = optimizer

    return optimizers


# Create buffers for each position and device
def create_buffers(
    rollout_length: int, num_buffers: int, device_iterator: Iterator[int]
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """Creates buffers for each position ('landlord', 'landlord_up',
    'landlord_down') and for each device (GPU or CPU). The buffers store
    tensors that will be used for experience replay during training.

    Args:
        rollout_length: The length of the rollout.
        num_buffers: The number of buffers to create.
        device_iterator: An iterable of device indices, typically GPU indices or 'cpu'.

    Returns:
        Dict[str, Buffers]: A dictionary where each device has a buffer of tensors for each position.
    """

    positions = ['landlord', 'landlord_up', 'landlord_down']
    buffers = {}

    for device in device_iterator:
        buffers[device] = {}

        for position in positions:
            # Set input dimension based on position
            feature_dim = 319 if position == 'landlord' else 430

            # Define the tensor specs for each buffer type
            specs = {
                'done':
                dict(size=(rollout_length, ), dtype=torch.bool),
                'episode_return':
                dict(size=(rollout_length, ), dtype=torch.float32),
                'target':
                dict(size=(rollout_length, ), dtype=torch.float32),
                'obs_x_no_action':
                dict(size=(rollout_length, feature_dim), dtype=torch.int8),
                'obs_action':
                dict(size=(rollout_length, 54), dtype=torch.int8),
                'obs_z':
                dict(size=(rollout_length, 5, 162), dtype=torch.int8),
            }

            # Initialize buffers for each spec
            position_buffers: Dict[str, List[torch.Tensor]] = {
                key: []
                for key in specs
            }

            for _ in range(num_buffers):
                # Create buffer tensors for each spec type
                for key, spec in specs.items():
                    if device != 'cpu':
                        buffer_tensor = (torch.empty(**spec).to(
                            torch.device(f'cuda:{device}')).share_memory_())
                    else:
                        buffer_tensor = (torch.empty(**spec).to(
                            torch.device('cpu')).share_memory_())

                    # Append the created buffer tensor to the corresponding buffer list
                    position_buffers[key].append(buffer_tensor)

            # Assign the buffers for this position to the device
            buffers[device][position] = position_buffers

    return buffers


def cards2tensor(list_cards: List[int]) -> torch.Tensor:
    """Converts a list of card integers to a 2D tensor representation. The
    tensor corresponds to the card's one-hot encoding as per the paper's
    format.

    The tensor is structured as a matrix where each row represents a specific card
    type (e.g., 3s, 4s, ..., Jokers), and the number of ones in the row indicates
    how many of those cards are present in the hand.

    Args:
        list_cards (List[int]): A list of integers representing cards,
                                where each integer maps to a specific card.

    Returns:
        torch.Tensor: A tensor of shape (4, 13), which is a one-hot encoded representation
                      of the card hand (matrix representation as described in the referenced paper).
    """
    # Convert list of cards to a numpy array following the _cards2array utility function
    matrix = _cards2array(list_cards)

    # Convert the numpy array to a torch tensor
    tensor = torch.from_numpy(
        matrix).int()  # Ensuring the tensor is integer-based

    return tensor


def act(
    env: gym.Env,
    worker_id: int,
    rollout_length: int,
    exp_epsilon: float,
    free_queue: Dict[str, mp.Queue],
    full_queue: Dict[str, mp.Queue],
    model: DouDiZhuModel,
    buffers: Dict[str, Dict[str, List[torch.Tensor]]],
    device: Union[str, int],
) -> None:
    """The actor process that interacts with the environment and fills buffers
    with data.

    The actor will run indefinitely until stopped, and it uses queues for synchronization
    with the main process. Data is transferred between the environment and the buffer
    for three positions in the game: 'landlord', 'landlord_up', and 'landlord_down'.

    Args:
        env: The environment to interact with
        worker_id: Process index for the actor.
        rollout_length: The length of the rollout.
        exp_epsilon: The exploration epsilon for the actor.
        free_queue: Queue for getting free buffer indices.
        full_queue: Queue for passing filled buffer indices to the main process.
        model: The model used for decision-making in the game.
        buffers: Shared memory buffers for storing game experiences.
        device: Device name ('cpu' or 'cuda:x') where this actor will run.
    """
    positions = ['landlord', 'landlord_up', 'landlord_down']

    env: EnvWrapper = EnvWrapper(env, device)
    try:
        # Initialize buffers for observations and episode returns
        done_buf = {p: deque(maxlen=rollout_length) for p in positions}
        episode_return_buf = {
            p: deque(maxlen=rollout_length)
            for p in positions
        }
        target_buf = {p: deque(maxlen=rollout_length) for p in positions}
        obs_x_no_action_buf = {
            p: deque(maxlen=rollout_length)
            for p in positions
        }
        obs_action_buf = {p: deque(maxlen=rollout_length) for p in positions}
        obs_z_buf = {p: deque(maxlen=rollout_length) for p in positions}
        size = {p: 0 for p in positions}

        # Initialize the environment
        position, obs, env_output = env.initial()

        while True:
            while True:
                # Collect data for the current position
                obs_x_no_action_buf[position].append(
                    env_output['obs_x_no_action'])
                obs_z_buf[position].append(env_output['obs_z'])

                with torch.no_grad():
                    agent_output = model.forward(
                        position,
                        obs['z_batch'],
                        obs['x_batch'],
                        training=False,
                        exp_epsilon=exp_epsilon,
                    )

                # Select and perform the action
                action_idx = int(agent_output['action'].cpu().detach().numpy())
                action = obs['legal_actions'][action_idx]
                obs_action_buf[position].append(cards2tensor(action))

                size[position] += 1

                # Step the environment with the chosen action
                position, obs, env_output = env.step(action)

                # Check if the game is done and update the buffers
                if env_output['done']:
                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            # Append episode termination status and returns
                            done_buf[p].extend([False] * (diff - 1))
                            done_buf[p].append(True)

                            episode_return = (env_output['episode_return']
                                              if p == 'landlord' else
                                              -env_output['episode_return'])
                            episode_return_buf[p].extend([0.0] * (diff - 1))
                            episode_return_buf[p].append(episode_return)
                            target_buf[p].extend([episode_return] * diff)
                    break

            # Move data to shared buffers when enough data is collected
            for p in positions:
                while size[p] >= rollout_length:
                    # Retrieve a free buffer index from the queue
                    index = free_queue[p].get()
                    if index is None:
                        break

                    # Transfer collected data to the shared buffer
                    for t in range(rollout_length):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = (
                            episode_return_buf[p][t])
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['obs_x_no_action'][index][t, ...] = (
                            obs_x_no_action_buf[p][t])
                        buffers[p]['obs_action'][index][
                            t, ...] = obs_action_buf[p][t]
                        buffers[p]['obs_z'][index][t, ...] = obs_z_buf[p][t]

                    # Notify that the buffer has been filled
                    full_queue[p].put(index)

                    # Remove transferred data from the local buffer
                    done_buf[p].popleft()
                    episode_return_buf[p].popleft()
                    target_buf[p].popleft()
                    obs_x_no_action_buf[p].popleft()
                    obs_action_buf[p].popleft()
                    obs_z_buf[p].popleft()
                    size[p] -= rollout_length

    except KeyboardInterrupt:
        logger.info('Actor %i stopped manually.', worker_id)
    except Exception as e:
        logger.error(f'Exception in worker process {worker_id}: {str(e)}')
        traceback.print_exc()
        raise e
