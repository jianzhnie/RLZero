from __future__ import print_function

from collections import deque
from typing import List, Tuple

import numpy as np


class Board(object):
    """board for the game.

    board states stored as a dict,
    key: move as location on the board,
    value: player as pieces type
    """

    def __init__(self,
                 width: int = 8,
                 height: int = 8,
                 n_in_row: int = 5) -> None:
        self.width = width
        self.height = height
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = n_in_row
        self.players = [1, 2]  # player1 and player2
        self.feature_planes = 4
        # how many binary feature planes we use,
        # in alphago zero is 17 and the input to the neural network is 19x19x17
        # here is a self.width x self.height x (self.feature_planes+1) binary feature planes,
        # the self.feature_planes is the number of history features
        # the additional plane is the color feature that indicate the current player
        # for example, in 11x11 board, is 11x11x9,8 for history features and 1 for current player
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)
        # use the deque to store last 8 moves
        # fill in with [-1,-1] when one game start to indicate no move

    def init_board(self, start_player: int = 0) -> None:
        """init the board and set some variables."""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception(
                f'Board width {self.width} and height {self.height} can not be less than {self.n_in_row}'
            )
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        # once a move has been played, remove it right away
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        self.states_sequence = deque(maxlen=self.feature_planes)
        self.states_sequence.extendleft([[-1, -1]] * self.feature_planes)

    def move_to_location(self, move: int) -> List:
        """3*3 board's moves like:

        6 7 8     3 4 5     0 1 2 and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location: List[int, int]) -> int:
        """From location to move.

        Args:
            location (List[int, int]): [x,y] x is the row, y is the column

        Returns:
            move: int: the move according to the location
        """

        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self) -> np.ndarray:
        """return the board state from the perspective of the current player.

        state shape: (self.feature_planes+1) x width x height
        """
        square_state = np.zeros(
            (self.feature_planes + 1, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # states contain the (key,value) indicate (move,player)
            # for example
            # self.states.items() get dict_items([(1, 1), (2, 1), (3, 2)])
            # zip(*) get [(1, 2, 3), (1, 1, 2)]
            # then np.array and get
            # moves = np.array([1, 2, 3])
            # players = np.array([1, 1, 2])
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            # to construct the binary feature planes as alphazero did
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[
                self.feature_planes][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move: int) -> None:
        """Update the board."""
        self.states[move] = self.current_player
        # save the move in states
        self.states_sequence.appendleft([move, self.current_player])
        # save the last some moves in deque，so as to construct the binary feature planes
        self.availables.remove(move)
        # remove the played move from self.availables
        self.current_player = (self.players[0] if self.current_player
                               == self.players[1] else self.players[1])
        # change the current player
        self.last_move = move

    def has_a_winner(self) -> Tuple[bool, int]:
        """Judge if there's a 5-in-a-row, and which player if so.

        Returns:
            Tuple[bool, int]: _description_
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        # 棋盘上所有棋子的位置
        moved = list(set(range(width * height)) - set(self.availables))
        # 当前所有棋子数量不足以获胜
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h, w = self.move_to_location(m)
            player = states[m]
            # 判断是否有水平线
            if (w in range(width - n + 1)
                    and len(set(states.get(i, -1)
                                for i in range(m, m + n))) == 1):
                return True, player
            # 判断是否有竖线
            if (h in range(height - n + 1) and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * width, width))) == 1):
                return True, player
            # 判断是否有斜线
            if (w in range(width - n + 1) and h in range(height - n + 1)
                    and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1)
                    and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self) -> Tuple[bool, int]:
        """Check whether the game is ended or not."""
        end, winner = self.has_a_winner()
        if end:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self) -> int:
        return self.current_player
