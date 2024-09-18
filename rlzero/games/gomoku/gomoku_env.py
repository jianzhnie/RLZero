from __future__ import print_function

from typing import List, Tuple
from uu import Error

import numpy as np

from ..base_env import BaseEnv


class GomokuEnv(BaseEnv):
    """board for the game.

    board states stored as a dict,
    key: move as location on the board,
    value: player as pieces type
    """

    def __init__(
        self,
        board_size: int = 8,
        n_in_row: int = 5,
        start_player_idx: int = 0,
    ) -> None:
        super().__init__()
        self.board_size = board_size
        self.n_in_row = n_in_row
        self.players = [0, 1]  # player1 and player2
        self.start_player_idx = start_player_idx  # start player
        self._current_player = self.players[self.start_player_idx]
        self._leagel_actions = list(range(self.board_size * self.board_size))

    def reset(self, start_player_idx: int = 0) -> None:
        """init the board and set some variables."""
        if self.board_size < self.n_in_row:
            raise Error(f'Board board_size can not less than {self.n_in_row}')
        if start_player_idx not in (0, 1):
            raise Error(
                f'{start_player_idx} should be 0 (player1 first) or 1 (player2 first)'
            )
        self.start_player_idx = start_player_idx
        self._current_player = self.players[start_player_idx]
        self._leagel_actions = list(range(self.board_size * self.board_size))
        self.states = {}
        self.last_move = -1
        self.info = {}
        return self.current_state()

    def step(self, action: int):
        """Update the board."""
        assert (action in self._leagel_actions), print(
            f'You input illegal action: {action}, the legal_actions are {self._leagel_actions}.'
        )

        self.states[action] = self._current_player
        self._leagel_actions.remove(action)
        # change the current player
        self.last_move = action
        win, winner = self.has_a_winner()
        reward = 0
        if win:
            if winner == self._current_player:
                reward = 1
            else:
                reward = -1

        self._current_player = (self.players[0] if self._current_player
                                == self.players[1] else self.players[1])
        obs = self.current_state()
        return obs, reward, win, self.info

    def leagel_actions(self):
        return self._leagel_actions

    def render(self):
        board_size = self.board_size
        p1, p2 = self.players
        print()
        for x in range(board_size):
            print('{0:8}'.format(x), end='')
        print('\r\n')
        for i in range(board_size - 1, -1, -1):
            print('{0:4d}'.format(i), end='')
            for j in range(board_size):
                loc = i * board_size + j
                p = self.states.get(loc, -1)
                if p == p1:
                    print('B'.center(8), end='')
                elif p == p2:
                    print('W'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def current_state(self) -> np.ndarray:
        """return the board state from the perspective of the current player.

        state shape: (self.feature_planes+1) x board_size x board_size
        """
        square_state = np.zeros((4, self.board_size, self.board_size))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self._current_player]
            move_oppo = moves[players != self._current_player]

            # to construct the binary feature planes as alphazero did
            square_state[0][tuple(self.move_to_location(move_curr))] = 1.0
            square_state[1][tuple(self.move_to_location(move_oppo))] = 1.0
            # indicate the last move location
            square_state[2][tuple(self.move_to_location(self.last_move))] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
            # indicate the colour to play
        return square_state

    def has_a_winner(self) -> Tuple[bool, int]:
        """Judge if there's a 5-in-a-row, and which player if so.

        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'is_win' and 'winner',
                - if player 1 win,     'is_win' = True, 'winner' = 1
                - if player 2 win,     'is_win' = True, 'winner' = 2
                - if draw,             'is_win' = False, 'winner' = -1
                - if game is not over, 'is_win' = False, 'winner' = -1
        """
        board_size = self.board_size
        states = self.states
        n = self.n_in_row
        # 棋盘上所有棋子的位置
        moved = list(
            set(range(board_size * board_size)) - set(self._leagel_actions))
        # 当前所有棋子数量不足以获胜
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h, w = self.move_to_location(m)
            player = states[m]
            # 判断是否有水平线
            if (w in range(board_size - n + 1)
                    and len(set(states.get(i, -1)
                                for i in range(m, m + n))) == 1):
                return True, player
            # 判断是否有竖线
            if (h in range(board_size - n + 1) and len(
                    set(
                        states.get(i, -1)
                        for i in range(m, m + n * board_size, board_size)))
                    == 1):
                return True, player
            # 判断是否有斜线
            if (w in range(board_size - n + 1)
                    and h in range(board_size - n + 1) and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (board_size + 1), board_size + 1)))
                    == 1):
                return True, player

            if (w in range(n - 1, board_size)
                    and h in range(board_size - n + 1) and len(
                        set(
                            states.get(i, -1)
                            for i in range(m, m + n *
                                           (board_size - 1), board_size - 1)))
                    == 1):
                return True, player

        return False, -1

    def get_done_reward(self):
        """
        Overview:
             Check if the game is over and what is the reward in the perspective of player 1.
             Return 'done' and 'reward'.
        Returns:
            - outputs (:obj:`Tuple`): Tuple containing 'done' and 'reward',
                - if player 1 win,     'done' = True, 'reward' = 1
                - if player 2 win,     'done' = True, 'reward' = -1
                - if draw,             'done' = True, 'reward' = 0
                - if game is not over, 'done' = False,'reward' = None
        """
        win, winner = self.has_a_winner()
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        elif winner == -1 and win:
            reward = 0
        elif winner == -1 and not win:
            # episode is not done
            reward = None
        return win, reward

    def game_end_winner(self):
        """Check whether the game is ended or not."""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self._leagel_actions):
            return True, -1
        return False, -1

    def is_terminal(self):
        """Check whether the game is ended or not."""
        game_end, winner = self.game_end_winner()
        return game_end

    def returns(self):
        """Check whether the game is ended or not.

        Total reward for each player over the course of the game so far.
        """
        win, winner = self.has_a_winner()

        if winner == 1:
            return [1, -1]
        elif winner == 2:
            return [-1, 1]
        elif winner == -1 and win:
            return [0, 0]
        elif winner == -1 and not win:
            return [0, 0]
        return [0, 0]

    def move_to_location(self, move: int) -> List:
        """3*3 board's moves like:

        6 7 8     3 4 5     0 1 2 and move 5's location is (1,2)
        """
        h = move // self.board_size
        w = move % self.board_size
        return [h, w]

    def location_to_move(self, location: List) -> int:
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
        move = h * self.board_size + w
        if move not in range(self.board_size * self.board_size):
            return -1
        return move

    def action_to_string(self, move: int):
        """
        Overview:
            Convert an action number to a string representing the action.
        Arguments:
            - action_number: an integer from the action space.
        Returns:
            - String representing the action.
        """
        row = move // self.board_size + 1
        col = move % self.board_size + 1
        return f'Play row {row}, column {col}'

    def max_utility(self):
        return 1

    def current_player(self):
        return self._current_player

    def legal_actions(self, player):
        return self._leagel_actions

    def current_player_index(self):
        """
        current_player_index = 0, current_player = 1
        current_player_index = 1, current_player = 2
        """
        return 0 if self._current_player == 1 else 1

    def __str__(self):
        return 'Gomoku Board'


if __name__ == '__main__':
    env = GomokuEnv()
    env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        print(obs)
        print(reward)
