import copy
from abc import ABC

import gymnasium


class BaseEnv(gymnasium.Env, ABC):

    def __init__(self):
        pass

    def render(self):
        raise NotImplementedError

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is
        over."""
        raise NotImplementedError

    def legal_actions(self, player):
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
