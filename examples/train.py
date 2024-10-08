import os
import sys

sys.path.append(os.getcwd())
from rlzero.algorithms.rl_args import parse_args
from RLZero.rlzero.algorithms.impala.impala_atari import ImpalaTrainer

if __name__ == '__main__':
    flags = parse_args()
    print(flags)
    agent = ImpalaTrainer(flags)
    agent.train()
