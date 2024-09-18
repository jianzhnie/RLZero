from typing import List

import numpy as np
import torch

from rlzero.envs.doudizhu.env import _cards2array
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
