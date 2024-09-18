import typing

import numpy as np
import torch

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

# Type alias for buffers
Buffers = typing.Dict[str, typing.List[torch.Tensor]]
