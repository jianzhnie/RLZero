# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An MCTS Evaluator for an AlphaZero model."""

import numpy as np
import pyspiel
from open_spiel.python.algorithms import mcts


class AlphaZeroEvaluator(mcts.Evaluator):
    """An AlphaZero MCTS Evaluator."""
    def __init__(self, game, agent, cache_size=2**16):
        """An AlphaZero MCTS Evaluator."""
        if game.num_players() != 2:
            raise ValueError('Game must be for two players.')
        game_type = game.get_type()
        if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
            raise ValueError('Game must have terminal rewards.')
        if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
            raise ValueError('Game must have sequential turns.')

        self._alphazeroAgent = agent

    def _inference(self, state):
        # Make a singleton batch
        obs = np.expand_dims(state.observation_tensor(), 0)
        mask = np.expand_dims(state.legal_actions_mask(), 0)
        value, policy = self._alphazeroAgent.predict(obs, mask)

        return value, policy  # Unpack batch

    def evaluate(self, state):
        """Returns a value for the given state."""
        value, _ = self._inference(state)
        return np.array([value, -value])

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        else:
            # Returns the probabilities for all actions.
            _, policy = self._inference(state)
            return [(action, policy[action])
                    for action in state.legal_actions()]
