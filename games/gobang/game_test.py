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

# Lint as python3
"""Tests for Python Tic-Tac-Toe."""

from absl.testing import absltest
import numpy as np
from open_spiel.python.algorithms.get_all_states import get_all_states
from game_openspiel import GomokuGame, GomokuGameState
from open_spiel.python.observation import make_observation
import pyspiel


class TicTacToeTest(absltest.TestCase):
    def test_can_create_game_and_state(self):
        """Checks we can create the game and a state."""
        game = GomokuGame()
        state = game.new_initial_state()
        self.assertEqual(str(state), "...\n...\n...")

    def test_random_game(self):
        """Tests basic API functions."""
        # This is here mostly to show the API by example.
        # More serious simulation tests are done in python/tests/games_sim_test.py
        # and in test_game_from_cc (below), both of which test the conformance to
        # the API thoroughly.
        game = GomokuGame()
        state = game.new_initial_state()
        while not state.is_terminal():
            print(state)
            cur_player = state.current_player()
            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions)
            print("Player {} chooses action {}".format(cur_player, action))
            state.apply_action(action)
        print(state)
        print("Returns: {}".format(state.returns()))

    def test_observation_tensors_same(self):
        """Checks observation tensor is the same from C++ and from Python."""
        game = GomokuGame()
        state = game.new_initial_state()
        for a in [4, 5, 2, 3]:
            state.apply_action(a)
        py_obs = make_observation(game)
        print("***", py_obs.tensor)
        py_obs.set_from(state, state.current_player())
        print("***", py_obs.tensor)
        cc_obs = state.observation_tensor()
        print("===", np.array(cc_obs))
        np.testing.assert_array_equal(py_obs.tensor, cc_obs)

    def test_cloned_state_matches_original_state(self):
        """Check we can clone states successfully."""
        game = GomokuGame()
        state = game.new_initial_state()
        state.apply_action(1)
        state.apply_action(2)
        clone = state.clone()

        self.assertEqual(state.history(), clone.history())
        self.assertEqual(state.num_players(), clone.num_players())
        self.assertEqual(state.move_number(), clone.move_number())
        self.assertEqual(state.num_distinct_actions(),
                         clone.num_distinct_actions())

        self.assertEqual(state._cur_player, clone._cur_player)
        self.assertEqual(state._player0_score, clone._player0_score)
        self.assertEqual(state._is_terminal, clone._is_terminal)
        np.testing.assert_array_equal(state.board, clone.board)


if __name__ == "__main__":
    absltest.main()
