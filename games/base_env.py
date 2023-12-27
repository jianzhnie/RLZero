import copy
from abc import ABC, abstractmethod
from typing import Any

import gymnasium


class BaseEnv(gymnasium.Env, ABC):

    def __init__(self):
        pass

    @abstractmethod
    def reset(self) -> Any:
        """
        Overview:
            Reset the env to an initial state and returns an initial observation.
        Returns:
            - obs (:obj:`Any`): Initial observation after reset.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> 'BaseEnv.timestep':
        """
        Overview:
            Run one timestep of the environment's dynamics/simulation.
        Arguments:
            - action (:obj:`Any`): The ``action`` input to step with.
        Returns:
            - timestep (:obj:`BaseEnv.timestep`): The result timestep of env executing one step.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self):
        raise NotImplementedError

    @abstractmethod
    def observation_space(self, agent):
        raise NotImplementedError

    @abstractmethod
    def action_space(self, agent):
        raise NotImplementedError

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is
        over."""
        raise NotImplementedError

    def legal_actions(self):
        """Returns a list of legal actions, sorted in ascending order."""
        raise NotImplementedError

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        raise NotImplementedError

    def clone(self):
        return copy.deepcopy(self)

    def is_terminal(self):
        """Returns True if the game is over."""
        raise NotImplementedError
