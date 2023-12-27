"""Adapt the Go environment in PettingZoo (https://github.com/Farama-
Foundation/PettingZoo) to the BaseEnv interface."""
from __future__ import annotations

import os
import sys
from collections import namedtuple

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import EzPickle
from pettingzoo import AECEnv
from pettingzoo.classic.go import coords, go_base
from pettingzoo.utils.agent_selector import agent_selector

BaseEnvTimestep = namedtuple('BaseEnvTimestep',
                             ['obs', 'reward', 'done', 'info'])


def get_image(path):
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(cwd + '/' + path)
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


class GoEnv(AECEnv, EzPickle):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'go_v5',
        'is_parallelizable': False,
        'render_fps': 2,
    }

    def __init__(
        self,
        board_size: int = 19,
        komi: float = 7.5,
        render_mode: str | None = None,
        screen_height: int | None = 800,
    ) -> None:
        EzPickle.__init__(self, board_size, komi, render_mode, screen_height)
        super().__init__()
        # board_size: a int, representing the board size (board has a board_size x board_size shape)
        # komi: a float, representing points given to the second player.
        self._overwrite_go_global_variables(board_size=board_size)
        self._komi = komi

        self.agents = ['black_0', 'white_0']
        self.possible_agents = self.agents[:]

        self.screen = None

        self.observation_spaces = self._convert_to_dict([
            spaces.Dict({
                'observation':
                spaces.Box(low=0,
                           high=1,
                           shape=(self._N, self._N, 17),
                           dtype=bool),
                'action_mask':
                spaces.Box(
                    low=0,
                    high=1,
                    shape=((self._N * self._N) + 1, ),
                    dtype=np.int8,
                ),
            }) for _ in range(self.num_agents)
        ])

        self.action_spaces = self._convert_to_dict([
            spaces.Discrete(self._N * self._N + 1)
            for _ in range(self.num_agents)
        ])

        self._agent_selector = agent_selector(self.agents)

        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)

        self.render_mode = render_mode
        self.screen_width = self.screen_height = screen_height

        if self.render_mode == 'human':
            self.clock = pygame.time.Clock()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _overwrite_go_global_variables(self, board_size: int):
        self._N = board_size
        go_base.N = self._N
        go_base.ALL_COORDS = [(i, j) for i in range(self._N)
                              for j in range(self._N)]
        go_base.EMPTY_BOARD = np.zeros([self._N, self._N], dtype=np.int8)
        go_base.NEIGHBORS = {(x, y): list(
            filter(self._check_bounds, [(x + 1, y), (x - 1, y), (x, y + 1),
                                        (x, y - 1)]))
                             for x, y in go_base.ALL_COORDS}
        go_base.DIAGONALS = {(x, y): list(
            filter(
                self._check_bounds,
                [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1),
                 (x - 1, y - 1)],
            ))
                             for x, y in go_base.ALL_COORDS}
        return

    def _check_bounds(self, c):
        return 0 <= c[0] < self._N and 0 <= c[1] < self._N

    def _encode_player_plane(self, agent):
        if agent == self.possible_agents[0]:
            return np.zeros([self._N, self._N], dtype=bool)
        else:
            return np.ones([self._N, self._N], dtype=bool)

    def _encode_board_planes(self, agent):
        agent_factor = (go_base.BLACK
                        if agent == self.possible_agents[0] else go_base.WHITE)
        current_agent_plane_idx = np.where(self._go.board == agent_factor)
        opponent_agent_plane_idx = np.where(self._go.board == -agent_factor)
        current_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        opponent_agent_plane = np.zeros([self._N, self._N], dtype=bool)
        current_agent_plane[current_agent_plane_idx] = 1
        opponent_agent_plane[opponent_agent_plane_idx] = 1
        return current_agent_plane, opponent_agent_plane

    def _int_to_name(self, ind):
        return self.possible_agents[ind]

    def _name_to_int(self, name):
        return self.possible_agents.index(name)

    def _convert_to_dict(self, list_of_list):
        return dict(zip(self.possible_agents, list_of_list))

    def _encode_legal_actions(self, actions):
        return np.where(actions == 1)[0]

    def _encode_rewards(self, result) -> list[int]:
        return [1, -1] if result == 1 else [-1, 1]

    @property
    def current_player(self):
        return self.current_player_index

    @property
    def to_play(self):
        return self.current_player_index

    def observe(self, agent):
        player_plane = self._encode_player_plane(agent)

        observation = np.dstack((self.board_history, player_plane))

        legal_moves = self.next_legal_moves if agent == self.agent_selection else []
        action_mask = np.zeros((self._N * self._N) + 1, 'int8')
        for i in legal_moves:
            action_mask[i] = 1

        return {'observation': observation, 'action_mask': action_mask}

    def step(self, action):
        if (self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]):
            return self._was_dead_step(action)
        self._go = self._go.play_move(coords.from_flat(action))
        self._last_obs = self.observe(self.agent_selection)
        current_agent_plane, opponent_agent_plane = self._encode_board_planes(
            self.agent_selection)
        self.board_history = np.dstack(
            (current_agent_plane, opponent_agent_plane,
             self.board_history[:, :, :-2]))
        next_player = self._agent_selector.next()

        current_agent = next_player  # 'black_0', 'white_0'
        current_index = self.agents.index(current_agent)  # 0, 1
        self.current_player_index = current_index

        if self._go.is_game_over():
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)])
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._go.result()))
            self.next_legal_moves = [self._N * self._N]
        else:
            self.next_legal_moves = self._encode_legal_actions(
                self._go.all_legal_moves())
        self.agent_selection = (next_player if next_player else
                                self._agent_selector.next())
        self._accumulate_rewards()

        if self.render_mode == 'human':
            self.render()

        # observation, reward, done, info = env.last()
        agent = self.agent_selection
        current_index = self.agents.index(agent)
        self.current_player_index = current_index
        observation = self.observe(agent)
        reward = self._cumulative_rewards[agent]
        done = self.terminations[agent]
        info = self.infos[agent]

        return BaseEnvTimestep(observation, reward, done, info)

    def reset(self, seed=None, options=None):
        self._go = go_base.Position(board=None, komi=self._komi)

        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._cumulative_rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.rewards = self._convert_to_dict(np.array([0.0, 0.0]))
        self.terminations = self._convert_to_dict(
            [False for _ in range(self.num_agents)])
        self.truncations = self._convert_to_dict(
            [False for _ in range(self.num_agents)])
        self.infos = self._convert_to_dict([{}
                                            for _ in range(self.num_agents)])
        self.next_legal_moves = self._encode_legal_actions(
            self._go.all_legal_moves())
        self._last_obs = self.observe(self.agents[0])
        self.board_history = np.zeros((self._N, self._N, 16), dtype=bool)
        self.current_player_index = 0

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                'You are calling render method without specifying any render mode.'
            )
            return

        if self.screen is None:
            pygame.init()

            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height))
                pygame.display.set_caption('Go')
            else:
                self.screen = pygame.Surface(
                    (self.screen_width, self.screen_height))

        size = go_base.N

        # Load and scale all of the necessary images
        tile_size = self.screen_width / size

        black_stone = get_image(os.path.join('img', 'GoBlackPiece.png'))
        black_stone = pygame.transform.scale(
            black_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        white_stone = get_image(os.path.join('img', 'GoWhitePiece.png'))
        white_stone = pygame.transform.scale(
            white_stone, (int(tile_size * (5 / 6)), int(tile_size * (5 / 6))))

        tile_img = get_image(os.path.join('img', 'GO_Tile0.png'))
        tile_img = pygame.transform.scale(
            tile_img, ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))

        # blit board tiles
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                self.screen.blit(tile_img,
                                 ((i * (tile_size)), int(j) * (tile_size)))

        for i in range(1, 9):
            tile_img = get_image(
                os.path.join('img', 'GO_Tile' + str(i) + '.png'))
            tile_img = pygame.transform.scale(
                tile_img,
                ((int(tile_size * (7 / 6))), int(tile_size * (7 / 6))))
            for j in range(1, size - 1):
                if i == 1:
                    self.screen.blit(tile_img, (0, int(j) * (tile_size)))
                elif i == 2:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)), 0))
                elif i == 3:
                    self.screen.blit(tile_img,
                                     ((size - 1) * (tile_size), int(j) *
                                      (tile_size)))
                elif i == 4:
                    self.screen.blit(tile_img, ((int(j) * (tile_size)),
                                                (size - 1) * (tile_size)))
            if i == 5:
                self.screen.blit(tile_img, (0, 0))
            elif i == 6:
                self.screen.blit(tile_img, ((size - 1) * (tile_size), 0))
            elif i == 7:
                self.screen.blit(tile_img, ((size - 1) * (tile_size),
                                            (size - 1) * (tile_size)))
            elif i == 8:
                self.screen.blit(tile_img, (0, (size - 1) * (tile_size)))

        offset = tile_size * (1 / 6)
        # Blit the necessary chips and their positions
        for i in range(0, size):
            for j in range(0, size):
                if self._go.board[i][j] == go_base.BLACK:
                    self.screen.blit(
                        black_stone,
                        ((i * (tile_size) + offset), int(j) *
                         (tile_size) + offset),
                    )
                elif self._go.board[i][j] == go_base.WHITE:
                    self.screen.blit(
                        white_stone,
                        ((i * (tile_size) + offset), int(j) *
                         (tile_size) + offset),
                    )

        if self.render_mode == 'human':
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (np.transpose(observation, axes=(1, 0, 2))
                if self.render_mode == 'rgb_array' else None)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {'legal_moves': []}

    def legal_actions(self):
        pass

    def legal_moves(self):
        if self._go.is_game_over():
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)])
            self.rewards = self._convert_to_dict(
                self._encode_rewards(self._go.result()))
            self.next_legal_moves = [self._N * self._N]
        else:
            self.next_legal_moves = self._encode_legal_actions(
                self._go.all_legal_moves())

        return self.next_legal_moves

    def random_action(self):
        action_list = self.legal_moves()
        return np.random.choice(action_list)

    def bot_action(self):
        # TODO
        pass

    def human_to_action(self):
        """
        Overview:
            For multiplayer games, ask the user for a legal action
            and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        # print(self.board)
        while True:
            try:
                print(
                    f'Current available actions for the player {self.to_play()} are:{self.legal_moves()}'
                )
                choice = int(
                    input(
                        f'Enter the index of next move for the player {self.to_play()}: '
                    ))
                if choice in self.legal_moves():
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception:
                print('Wrong input, try again')
        return choice

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return 'LightZero Go Env'
