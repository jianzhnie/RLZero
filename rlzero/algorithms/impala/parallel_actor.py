import os
import pprint
import threading
import timeit
import traceback
from collections import deque
from queue import Queue
from typing import Dict, Iterator, List, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo import AECEnv
from torch import multiprocessing as mp
from torch.optim import Optimizer

from rlzero.algorithms.rl_args import RLArguments
from rlzero.utils.logger_utils import get_logger

logger = get_logger("rlzero")


class ImpalaTrainer(object):
    """The ImpalaTrainer class is responsible for managing the training process of the Impala algorithm.

    This class handles:
    - Creating buffers to store experience data.
    - Using multi-threading for batch processing and learning.
    - Periodically saving checkpoints.
    - Logging and outputting training statistics.
    """

    def __init__(
        self,
        env: gym.Env,
        args: RLArguments = RLArguments,
    ) -> None:
        """Initialize the DistributedDouZero system.

        Args:
            args: Configuration arguments for the training process.
        """
        self.env = env

        self.args: RLArguments = args
