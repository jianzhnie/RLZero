[1mdiff --git a/games/__init__.py b/games/__init__.py[m
[1mindex e69de29..2831f6d 100644[m
[1m--- a/games/__init__.py[m
[1m+++ b/games/__init__.py[m
[36m@@ -0,0 +1,6 @@[m
[32m+[m[32mfrom .base_env import BaseEnv[m
[32m+[m[32mfrom .gomoku.gomoku_env import GomokuEnv[m
[32m+[m[32mfrom .gomoku.game import GameControl[m
[32m+[m
[32m+[m
[32m+[m[32m__all__ = ["BaseEnv", "GomokuEnv", "GameControl"][m
\ No newline at end of file[m
[1mdiff --git a/games/gomoku/alphazero_agent.py b/games/gomoku/alphazero_agent.py[m
[1mindex 0a5daa5..9a54fc8 100644[m
[1m--- a/games/gomoku/alphazero_agent.py[m
[1m+++ b/games/gomoku/alphazero_agent.py[m
[36m@@ -50,7 +50,7 @@[m [mclass AlphaZeroAgent(object):[m
         input: a batch of states[m
         output: a batch of action probabilities and state values[m
         """[m
[31m-        state_batch = torch.FloatTensor(state_batch).to(self.device)[m
[32m+[m[32m        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)[m
         log_act_probs, value = self.policy_value_net(state_batch)[m
         act_probs = np.exp(log_act_probs.detach().cpu().numpy())[m
         value = value.detach().cpu().numpy()[m
[36m@@ -60,9 +60,9 @@[m [mclass AlphaZeroAgent(object):[m
         """perform a training step."""[m
         # train mode[m
         self.policy_value_net.train()[m
[31m-        state_batch = torch.FloatTensor(state_batch).to(self.device)[m
[31m-        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)[m
[31m-        target_batch = torch.FloatTensor(target_vs).to(self.device)[m
[32m+[m[32m        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)[m
[32m+[m[32m        mcts_probs = torch.FloatTensor(np.array(mcts_probs)).to(self.device)[m
[32m+[m[32m        target_batch = torch.FloatTensor(np.array(target_vs)).to(self.device)[m
 [m
         log_act_probs, value = self.policy_value_net(state_batch)[m
 [m
[1mdiff --git a/tools/train.py b/tools/train.py[m
[1mdeleted file mode 100644[m
[1mindex dc601b9..0000000[m
[1m--- a/tools/train.py[m
[1m+++ /dev/null[m
[36m@@ -1,196 +0,0 @@[m
[31m-from __future__ import print_function[m
[31m-[m
[31m-import random[m
[31m-import sys[m
[31m-from collections import defaultdict, deque[m
[31m-[m
[31m-import numpy as np[m
[31m-import torch[m
[31m-[m
[31m-sys.path.append('..')[m
[31m-from games.game import Game[m
[31m-from games.gomoku.alphazero_agent import AlphaZeroAgent[m
[31m-from games.gomoku.gomoku_env import GomokuEnv[m
[31m-from rlzero.mcts.alphazero_mcts import AlphaZeroPlayer[m
[31m-from rlzero.mcts.rollout_mcts import RolloutPlayer[m
[31m-[m
[31m-[m
[31m-class TrainPipeline:[m
[31m-[m
[31m-    def __init__(self):[m
[31m-        # params of the board and the game[m
[31m-        self.board_size = 6[m
[31m-        self.n_in_row = 4[m
[31m-        self.board = GomokuEnv(board_size=self.board_size,[m
[31m-                               n_in_row=self.n_in_row)[m
[31m-        self.game = Game(self.board)[m
[31m-        # training params[m
[31m-        self.learn_rate = 2e-3[m
[31m-        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL[m
[31m-        self.temperature = 1.0  # the temperature param[m
[31m-        self.n_playout = 400  # num of simulations for each move[m
[31m-        self.c_puct = 5[m
[31m-        self.buffer_size = 1000[m
[31m-        self.batch_size = 32  # mini-batch size for training[m
[31m-        self.data_buffer = deque(maxlen=self.buffer_size)[m
[31m-        self.play_batch_size = 1[m
[31m-        self.epochs = 5  # num of train_steps for each update[m
[31m-        self.kl_targ = 0.02[m
[31m-        self.check_freq = 50[m
[31m-        self.game_batch_num = 64[m
[31m-        self.best_win_ratio = 0.0[m
[31m-        self.device = (torch.device('cuda')[m
[31m-                       if torch.cuda.is_available() else torch.device('cpu'))[m
[31m-[m
[31m-        # num of simulations used for the pure mcts, which is used as[m
[31m-        # the opponent to evaluate the trained policy[m
[31m-        self.pure_mcts_playout_num = 100[m
[31m-        self.alphazero_agent: AlphaZeroAgent = AlphaZeroAgent([m
[31m-            self.board_size,[m
[31m-            device=self.device,[m
[31m-        )[m
[31m-[m
[31m-        self.mcts_player = AlphaZeroPlayer([m
[31m-            self.alphazero_agent.policy_value_fn,[m
[31m-            n_playout=self.n_playout,[m
[31m-            c_puct=self.c_puct,[m
[31m-            is_selfplay=True,[m
[31m-        )[m
[31m-[m
[31m-    def get_equi_data(self, play_data):[m
[31m-        """augment the data set by rotation and flipping[m
[31m-        play_data: [(state, mcts_prob, winner_z), ..., ...][m
[31m-        """[m
[31m-        extend_data = [][m
[31m-        for state, mcts_porb, winner in play_data:[m
[31m-            for i in [1, 2, 3, 4]:[m
[31m-                # rotate counterclockwise[m
[31m-                equi_state = np.array([np.rot90(s, i) for s in state])[m
[31m-                equi_mcts_prob = np.rot90([m
[31m-                    np.flipud([m
[31m-                        mcts_porb.reshape(self.board_size, self.board_size)),[m
[31m-                    i)[m
[31m-                extend_data.append([m
[31m-                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))[m
[31m-                # flip horizontally[m
[31m-                equi_state = np.array([np.fliplr(s) for s in equi_state])[m
[31m-                equi_mcts_prob = np.fliplr(equi_mcts_prob)[m
[31m-                extend_data.append([m
[31m-                    (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))[m
[31m-        return extend_data[m
[31m-[m
[31m-    def collect_selfplay_data(self, n_games=1):[m
[31m-        """collect self-play data for training."""[m
[31m-        for i in range(n_games):[m
[31m-            winner, play_data = self.game.start_self_play([m
[31m-                self.mcts_player, temperature=self.temperature)[m
[31m-            play_data = list(play_data)[:][m
[31m-            self.episode_len = len(play_data)[m
[31m-            # augment the data[m
[31m-            play_data = self.get_equi_data(play_data)[m
[31m-            self.data_buffer.extend(play_data)[m
[31m-[m
[31m-    def policy_update(self):[m
[31m-        """update the policy-value net."""[m
[31m-        mini_batch = random.sample(self.data_buffer, self.batch_size)[m
[31m-        state_batch = [data[0] for data in mini_batch][m
[31m-        mcts_probs_batch = [data[1] for data in mini_batch][m
[31m-        winner_batch = [data[2] for data in mini_batch][m
[31m-        old_probs, old_v = self.alphazero_agent.policy_value(state_batch)[m
[31m-        for i in range(self.epochs):[m
[31m-            loss, entropy = self.alphazero_agent.learn(state_batch,[m
[31m-                                                       mcts_probs_batch,[m
[31m-                                                       winner_batch)[m
[31m-            new_probs, new_v = self.alphazero_agent.policy_value(state_batch)[m
[31m-            kl = np.mean([m
[31m-                np.sum([m
[31m-                    old_probs *[m
[31m-                    (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),[m
[31m-                    axis=1,[m
[31m-                ))[m
[31m-            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly[m
[31m-                break[m
[31m-        # adaptively adjust the learning rate[m
[31m-        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:[m
[31m-            self.lr_multiplier /= 1.5[m
[31m-        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:[m
[31m-            self.lr_multiplier *= 1.5[m
[31m-[m
[31m-        explained_var_old = 1 - np.var([m
[31m-            np.array(winner_batch) - old_v.flatten()) / np.var([m
[31m-                np.array(winner_batch))[m
[31m-        explained_var_new = 1 - np.var([m
[31m-            np.array(winner_batch) - new_v.flatten()) / np.var([m
[31m-                np.array(winner_batch))[m
[31m-        print(('kl:{:.5f},'[m
[31m-               'lr_multiplier:{:.3f},'[m
[31m-               'loss:{},'[m
[31m-               'entropy:{},'[m
[31m-               'explained_var_old:{:.3f},'[m
[31m-               'explained_var_new:{:.3f}').format([m
[31m-                   kl,[m
[31m-                   self.lr_multiplier,[m
[31m-                   loss,[m
[31m-                   entropy,[m
[31m-                   explained_var_old,[m
[31m-                   explained_var_new,[m
[31m-               ))[m
[31m-        return loss, entropy[m
[31m-[m
[31m-    def policy_evaluate(self, n_games=10):[m
[31m-        """[m
[31m-        Evaluate the trained policy by playing against the pure MCTS player[m
[31m-        Note: this is only for monitoring the progress of training[m
[31m-        """[m
[31m-        current_mcts_player = AlphaZeroPlayer([m
[31m-            self.alphazero_agent.policy_value_fn,[m
[31m-            n_playout=self.n_playout,[m
[31m-            c_puct=self.c_puct,[m
[31m-        )[m
[31m-        pure_mcts_player = RolloutPlayer([m
[31m-            n_playout=self.pure_mcts_playout_num,[m
[31m-            c_puct=5,[m
[31m-        )[m
[31m-        win_cnt = defaultdict(int)[m
[31m-        for i in range(n_games):[m
[31m-            winner = self.game.start_play(current_mcts_player,[m
[31m-                                          pure_mcts_player,[m
[31m-                                          start_player=i % 2,[m
[31m-                                          is_shown=0)[m
[31m-            win_cnt[winner] += 1[m
[31m-        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games[m
[31m-        print('num_playouts:{}, win: {}, lose: {}, tie:{}'.format([m
[31m-            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))[m
[31m-        return win_ratio[m
[31m-[m
[31m-    def run(self):[m
[31m-        """run the training pipeline."""[m
[31m-        try:[m
[31m-            for i in range(self.game_batch_num):[m
[31m-                self.collect_selfplay_data(self.play_batch_size)[m
[31m-                print('batch i:{}, episode_len:{}'.format([m
[31m-                    i + 1, self.episode_len))[m
[31m-                if len(self.data_buffer) > self.batch_size:[m
[31m-                    loss, entropy = self.policy_update()[m
[31m-                # check the performance of the current model,[m
[31m-                # and save the model params[m
[31m-                if (i + 1) % self.check_freq == 0:[m
[31m-                    print('current self-play batch: {}'.format(i + 1))[m
[31m-                    win_ratio = self.policy_evaluate()[m
[31m-                    self.alphazero_agent.save_model('./current_policy.model')[m
[31m-                    if win_ratio > self.best_win_ratio:[m
[31m-                        print('New best policy!!!!!!!!')[m
[31m-                        self.best_win_ratio = win_ratio[m
[31m-                        # update the best_policy[m
[31m-                        self.alphazero_agent.save_model('./best_policy.model')[m
[31m-                        if (self.best_win_ratio == 1.0[m
[31m-                                and self.pure_mcts_playout_num < 5000):[m
[31m-                            self.pure_mcts_playout_num += 1000[m
[31m-                            self.best_win_ratio = 0.0[m
[31m-        except KeyboardInterrupt:[m
[31m-            print('\n\rquit')[m
[31m-[m
[31m-[m
[31m-if __name__ == '__main__':[m
[31m-    training_pipeline = TrainPipeline()[m
[31m-    training_pipeline.run()[m
