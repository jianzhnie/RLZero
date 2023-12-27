from __future__ import print_function

from typing import List, Tuple
from uu import Error

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete


class GomokuEnv(gymnasium.Env):
    """board for the game.

    board states stored as a dict,
    key: move as location on the board,
    value: player as pieces type
    """

    def __init__(
        self,
        width: int = 8,
        height: int = 8,
        n_in_row: int = 5,
        start_player_idx: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        self.players = [1, 2]  # player1 and player2
        self.start_player_idx = start_player_idx  # start player
        self._current_player = self.players[self.start_player_idx]
        self.action_space = Discrete(width * height)
        self.observation_space = Box(0, 1, shape=(4, width, height))
        self._leagel_actions = list(range(self.width * self.height))

    def reset(self, start_player_idx: int = 0) -> None:
        """init the board and set some variables."""
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Error(
                f'Board width and height can not less than {self.n_in_row}')
        self.start_player_idx = start_player_idx
        self._current_player = self.players[start_player_idx]
        # start player
        # keep available moves in a list
        # once a move has been played, remove it right away
        self._leagel_actions = list(range(self.width * self.height))
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
        done, winner = self.get_done_winner()
        reward = 0
        if done:
            if winner == self._current_player:
                reward = 1
            else:
                reward = -1

        self._current_player = (self.players[0] if self._current_player
                                == self.players[1] else self.players[1])
        obs = self.current_state()
        return obs, reward, done, self.info

    def leagel_actions(self):
        return self._leagel_actions

    def render(self):
        width = self.width
        height = self.height
        p1, p2 = self.players
        print()
        for x in range(width):
            print('{0:8}'.format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print('{0:4d}'.format(i), end='')
            for j in range(width):
                loc = i * width + j
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

        state shape: (self.feature_planes+1) x width x height
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            # states contain the (key,value) indicate (move,player)
            # for example
            # self.states.items() get dict_items([(1, 1), (2, 1), (3, 2)])
            # zip(*) get [(1, 2, 3), (1, 1, 2)]
            # then np.array and get
            # moves = np.array([1, 2, 3])
            # players = np.array([1, 1, 2])
            move_curr = moves[players == self._current_player]
            move_oppo = moves[players != self._current_player]

            # to construct the binary feature planes as alphazero did
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0
            # indicate the colour to play
        return square_state[:, ::-1, :]

    def get_done_winner(self) -> Tuple[bool, int]:
        """Judge if there's a 5-in-a-row, and which player if so.

        Returns:
            Tuple[bool, int]: _description_
        """
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row
        # 棋盘上所有棋子的位置
        moved = list(set(range(width * height)) - set(self._leagel_actions))
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
        done, winner = self.get_done_winner()
        if winner == 1:
            reward = 1
        elif winner == 2:
            reward = -1
        elif winner == -1 and done:
            reward = 0
        elif winner == -1 and not done:
            # episode is not done
            reward = None
        return done, reward

    def game_end_winner(self):
        """Check whether the game is ended or not."""
        done, winner = self.get_done_winner()
        if done:
            return True, winner
        elif not len(self._leagel_actions):
            return True, -1
        return False, -1

    def move_to_location(self, move: int) -> List:
        """3*3 board's moves like:

        6 7 8     3 4 5     0 1 2 and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
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
        move = h * self.width + w
        if move not in range(self.width * self.height):
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
        row = move // self.width + 1
        col = move % self.width + 1
        return f'Play row {row}, column {col}'

    def current_player(self):
        return self._current_player

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
