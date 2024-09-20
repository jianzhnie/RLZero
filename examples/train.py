import os
import sys

sys.path.append(os.getcwd())
from rlzero.agents.dmc import DistributedDouZero
from rlzero.agents.rl_args import parse_args

if __name__ == '__main__':
    flags = parse_args()
    print(flags)
    agent = DistributedDouZero(flags)
    agent.train()
