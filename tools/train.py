from __future__ import print_function

import random
import sys
from collections import defaultdict, deque

import numpy as np
import torch

sys.path.append('..')
from games.game import Game
from games.gomoku.alphazero_agent import AlphaZeroAgent
from games.gomoku.gomoku_env import GomokuEnv
from rlzero.mcts.alphazero_player import AlphaZeroPlayer
from rlzero.mcts.rollout_player import RolloutPlayer


class TrainPipeline:

    def __init__(self):
        # params of the board and the game
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = GomokuEnv(width=self.board_width,
                               height=self.board_height,
                               n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temperature = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 1000
        self.batch_size = 32  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 64
        self.best_win_ratio = 0.0
        self.device = (torch.device('cuda')
                       if torch.cuda.is_available() else torch.device('cpu'))

        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 100
        self.alphazero_agent: AlphaZeroAgent = AlphaZeroAgent(
            self.board_width,
            self.board_height,
            device=self.device,
        )

        self.mcts_player = AlphaZeroPlayer(
            self.alphazero_agent.policy_value_fn,
            n_playout=self.n_playout,
            c_puct=self.c_puct,
            is_selfplay=True,
        )

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(
                        mcts_porb.reshape(self.board_height,
                                          self.board_width)), i)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append(
                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training."""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(
                self.mcts_player, temperature=self.temperature)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net."""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.alphazero_agent.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.alphazero_agent.learn(state_batch,
                                                       mcts_probs_batch,
                                                       winner_batch)
            new_probs, new_v = self.alphazero_agent.policy_value(state_batch)
            kl = np.mean(
                np.sum(
                    old_probs *
                    (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10))))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(
            np.array(winner_batch) - old_v.flatten()) / np.var(
                np.array(winner_batch))
        explained_var_new = 1 - np.var(
            np.array(winner_batch) - new_v.flatten()) / np.var(
                np.array(winner_batch))
        print(('kl:{:.5f},'
               'lr_multiplier:{:.3f},'
               'loss:{},'
               'entropy:{},'
               'explained_var_old:{:.3f},'
               'explained_var_new:{:.3f}').format(
                   kl,
                   self.lr_multiplier,
                   loss,
                   entropy,
                   explained_var_old,
                   explained_var_new,
               ))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = AlphaZeroPlayer(
            self.alphazero_agent.policy_value_fn,
            n_playout=self.n_playout,
            c_puct=self.c_puct,
        )
        pure_mcts_player = RolloutPlayer(
            n_playout=self.pure_mcts_playout_num,
            c_puct=5,
        )
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print('num_playouts:{}, win: {}, lose: {}, tie:{}'.format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline."""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print('batch i:{}, episode_len:{}'.format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print('current self-play batch: {}'.format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.alphazero_agent.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print('New best policy!!!!!!!!')
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.alphazero_agent.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0
                                and self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
