import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append(os.getcwd())
from rlzero.algorithms.impala.impala_dqn import ImpalaDQN

if __name__ == "__main__":
    impala_dqn = ImpalaDQN(state_dim=4, action_dim=2)
    impala_dqn.run()
