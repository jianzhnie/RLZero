import numpy as np
import pyspiel
from open_spiel.python.observation import IIGObserverForPublicInfoGame

_NUM_PLAYERS = 2
_NUM_ROWS = 8
_NUM_COLS = 8
_N_IN_ROW = 5
_NUM_CELLS = _NUM_ROWS * _NUM_COLS

_GAME_TYPE = pyspiel.GameType(
    short_name='python_gomoku',
    long_name='Python Gomoku',
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})
_GAME_INFO = pyspiel.GameInfo(num_distinct_actions=_NUM_CELLS,
                              max_chance_outcomes=0,
                              num_players=2,
                              min_utility=-1.0,
                              max_utility=1.0,
                              utility_sum=0.0,
                              max_game_length=_NUM_CELLS)


class GomokuGame(pyspiel.Game):
    """A Python version of the Tic-Tac-Toe game."""
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        self.observation_shape = (_NUM_PLAYERS + 2, _NUM_ROWS, _NUM_COLS)

    def observation_tensor_shape(self):
        return self.observation_shape

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return GomokuGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        if params is None:
            params = dict()

        params['observation_shape'] = self.observation_shape
        return BoardObserver(params)


class GomokuGameState(pyspiel.State):
    """A python version of the Tic-Tac-Toe state."""
    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        super().__init__(game)
        self._cur_player = 0
        self._player0_score = 0.0
        self._is_terminal = False
        self.num_rows = _NUM_ROWS
        self.num_cols = _NUM_COLS
        self.n_in_row = _N_IN_ROW
        self.states = {}
        self.last_move = -1
        self.board = np.full((self.num_rows, self.num_cols), '.')
        self.observation = np.zeros((4, self.num_rows, self.num_cols))

    # OpenSpiel (PySpiel) API functions are below. This is the standard set that
    # should be implemented by every perfect-information sequential-move game.

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is
        over."""
        return pyspiel.PlayerId.TERMINAL if self._is_terminal else self._cur_player

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [a for a in range(_NUM_CELLS) if self.board[_coord(a)] == '.']

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        self.board[_coord(action)] = 'x' if self._cur_player == 0 else 'o'
        self.states[action] = self._cur_player
        self.last_move = action
        self.observation = self.observation_tensor()  # Store this for later

        # if has a winner
        if self.has_a_winner():
            self._is_terminal = True
            self._player0_score = 1.0 if self._cur_player == 0 else -1.0
        elif all(self.board.ravel() != '.'):
            self._is_terminal = True
        else:
            self._cur_player = 1 - self._cur_player

    def observation_tensor(self):
        """return the board state from the perspective of the current player.

        state shape: 4*width*height
        """
        square_state = np.zeros((4, self.num_rows, self.num_cols), np.float32)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self._cur_player]
            move_oppo = moves[players != self._cur_player]
            # indicate the current move of player_1 and player_2
            square_state[0][_coord(move_curr)] = 1.0
            square_state[1][_coord(move_oppo)] = 1.0

            # indicate the last move location
            square_state[2][_coord(self.last_move)] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play

        return np.array(square_state)

    def has_a_winner(self):
        width = self.num_rows
        height = self.num_cols
        states = self.states
        n = self.n_in_row

        moved = list(
            set(range(width * height)) -
            set(self._legal_actions(player=self._cur_player)))
        if len(moved) < self.n_in_row * 2 - 1:
            return False

        for m in moved:
            h = m // width
            w = m % width
            if (w in range(width - n + 1)
                    and len(set(states.get(i, -1)
                                for i in range(m, m + n))) == 1):
                return True

            if (h in range(height - n + 1) and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * width, width))) == 1):
                return True

            if (w in range(width - n + 1) and h in range(height - n + 1)
                    and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (width + 1), width + 1))) == 1):
                return True

            if (w in range(n - 1, width) and h in range(height - n + 1)
                    and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (width - 1), width - 1))) == 1):
                return True

        return False

    def _action_to_string(self, player, action):
        """Action -> string."""
        row, col = _coord(action)
        return '{}({},{})'.format('x' if player == 0 else 'o', row, col)

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._is_terminal

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        return [self._player0_score, -self._player0_score]

    def __str__(self):
        """String for debug purposes.

        No particular semantics are required.
        """
        return _board_to_string(self.board)


class BoardObserver:
    """Observer, conforming to the PyObserver interface (see
    observation.py)."""
    def __init__(self, params):
        """Initializes an empty observation tensor."""
        # The observation should contain a 1-D tensor in `self.tensor` and a
        # dictionary of views onto the tensor, which may be of any shape.
        # Here the observation is indexed `(cell state, row, column)`.
        shape = params['observation_shape']
        self.tensor = np.zeros(np.prod(shape), np.float32)
        self.dict = {'observation': np.reshape(self.tensor, shape)}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of
        `player`."""
        # We update the observation via the shaped tensor since indexing is more
        # convenient than with the 1-D tensor. Both are views onto the same memory.
        self.tensor.fill(0)
        if 'observation' in self.dict:
            self.dict['observation'] = state.observation

    def string_from(self, state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        return _board_to_string(state.board)


# Helper functions for game details.


def _line_value(line):
    """Checks a possible line, returning the winning symbol if any."""
    if all(line == 'x') or all(line == 'o'):
        return line[0]


def _line_exists(board):
    """Checks if a line exists, returns "x" or "o" if so, and None
    otherwise."""
    return (_line_value(board[0]) or _line_value(board[1])
            or _line_value(board[2]) or _line_value(board[:, 0])
            or _line_value(board[:, 1]) or _line_value(board[:, 2])
            or _line_value(board.diagonal())
            or _line_value(np.fliplr(board).diagonal()))


def _coord(move):
    """Returns (row, col) from an action id."""
    return (move // _NUM_COLS, move % _NUM_COLS)


def _board_to_string(board):
    """Returns a string representation of the board."""
    return '\n'.join(''.join(row) for row in board)
