import collections
import random

import numpy as np


class TrainInput(
        collections.namedtuple('TrainInput',
                               'observation legals_mask policy value')):
    """Inputs for training the Model."""
    @staticmethod
    def stack(train_inputs):
        observation, legals_mask, policy, value = zip(*train_inputs)
        return TrainInput(np.array(observation, dtype=np.float32),
                          np.array(legals_mask, dtype=np.bool),
                          np.array(policy), np.expand_dims(value, 1))


class TrajectoryState(object):
    """A particular point along a trajectory."""
    def __init__(self, observation, current_player, legals_mask, action,
                 policy, value):
        self.observation = observation
        self.current_player = current_player
        self.legals_mask = legals_mask
        self.action = action
        self.policy = policy
        self.value = value


class Trajectory(object):
    """A sequence of observations, actions and policies, and the outcomes."""
    def __init__(self):
        self.states = []
        self.returns = None

    def add(self, information_state, action, policy):
        self.states.append((information_state, action, policy))


class Buffer(object):
    """A fixed size buffer that keeps the newest values."""
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = []
        self.total_seen = 0  # The number of items that have passed through.

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        return bool(self.data)

    def append(self, val):
        return self.extend([val])

    def extend(self, batch):
        batch = list(batch)
        self.total_seen += len(batch)
        self.data.extend(batch)
        self.data[:-self.max_size] = []

    def sample(self, count):
        return random.sample(self.data, count)
