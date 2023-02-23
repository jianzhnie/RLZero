from __future__ import print_function

import sys
from collections import defaultdict

import numpy as np
from alphazero_agent import AlphaZeroAgent
from config import AlphaZeroConfig
from game import GomokuGame

sys.path.append('../../')
from replaybuffer import Buffer

from muzero.mcts.mcts_alphazero import MCTSPlayer
from muzero.mcts.mcts_pure import MCTSPlayer as MCTS_Pure


def get_equi_data(play_data, board_width, board_height):
    """augment the data set by rotation and flipping
    play_data: [(state, mcts_prob, winner_z), ..., ...]
    """
    extend_data = []
    for state, mcts_porb, winner in play_data:
        for i in [1, 2, 3, 4]:
            # rotate counterclockwise
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(mcts_porb.reshape(board_height, board_width)), i)
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
            # flip horizontally
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append(
                (equi_state, np.flipud(equi_mcts_prob).flatten(), winner))
    return extend_data


def collect_selfplay_data(game: GomokuGame,
                          replay_buffer: Buffer,
                          mcts_player: MCTSPlayer,
                          config: AlphaZeroConfig,
                          n_games=1):
    """collect config-play data for training."""
    for i in range(n_games):
        winner, play_data = game.start_self_play(
            mcts_player, temperature=config.temperature)
        play_data = list(play_data)[:]
        replay_buffer.extend(play_data)
        config.episode_len = len(play_data)
        # augment the data
        play_data = get_equi_data(play_data, config.board_width,
                                  config.board_height)
        replay_buffer.extend(play_data)

    return replay_buffer


def policy_update(agent: AlphaZeroAgent, buffer: Buffer,
                  config: AlphaZeroConfig):
    """update the policy-value net."""

    mini_batch = buffer.sample(config.batch_size)
    state_batch = [data[0] for data in mini_batch]
    mcts_probs_batch = [data[1] for data in mini_batch]
    winner_batch = [data[2] for data in mini_batch]
    old_probs, old_v = agent.predict(state_batch)
    for i in range(config.epochs):
        loss, policy_loss, value_loss, entropy = agent.learn(
            state_batch, mcts_probs_batch, winner_batch)
        new_probs, new_v = agent.predict(state_batch)
        kl = np.mean(
            np.sum(old_probs *
                   (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                   axis=1))
        if kl > config.kl_targ * 4:  # early stopping if D_KL diverges badly
            break
    # adaptively adjust the learning rate
    if kl > config.kl_targ * 2 and config.lr_multiplier > 0.1:
        config.lr_multiplier /= 1.5
    elif kl < config.kl_targ / 2 and config.lr_multiplier < 10:
        config.lr_multiplier *= 1.5

    explained_var_old = (1 - np.var(np.array(winner_batch) - old_v.flatten()) /
                         np.var(np.array(winner_batch)))
    explained_var_new = (1 - np.var(np.array(winner_batch) - new_v.flatten()) /
                         np.var(np.array(winner_batch)))
    print(('kl:{:.5f},'
           'lr_multiplier:{:.3f},'
           'loss:{},'
           'entropy:{},'
           'explained_var_old:{:.3f},'
           'explained_var_new:{:.3f}').format(kl, config.lr_multiplier, loss,
                                              entropy, explained_var_old,
                                              explained_var_new))
    return loss, entropy


def policy_evaluate(agent: AlphaZeroAgent, game: GomokuGame,
                    config: AlphaZeroConfig):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(agent.policy_value_fn,
                                     c_puct=config.c_puct,
                                     n_playout=config.n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5,
                                 n_playout=config.pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    for i in range(config.n_games):
        winner = game.start_play(current_mcts_player,
                                 pure_mcts_player,
                                 start_player=i % 2,
                                 is_shown=False)
        win_cnt[winner] += 1
    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / config.n_games
    print('num_playouts:{}, win: {}, lose: {}, tie:{}'.format(
        config.n_games, win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


def main():
    config = AlphaZeroConfig()
    gomokugame = GomokuGame(width=config.board_width,
                            height=config.board_height,
                            n_in_row=config.n_in_row)
    alphazero = AlphaZeroAgent(config.board_width,
                               config.board_height,
                               learning_rate=config.learning_rate,
                               weight_decay=config.weight_decay,
                               device=config.device)
    mcts_player = MCTSPlayer(alphazero.policy_value_fn,
                             c_puct=config.c_puct,
                             n_playout=config.n_playout,
                             is_selfplay=True)
    rpm = Buffer(max_size=config.replay_buffer_size)
    for i in range(config.episode_size):
        winner, play_data = gomokugame.start_self_play(
            mcts_player, temperature=config.temperature)
        play_data = list(play_data)[:]

        rpm.extend(play_data)
        # augment the data
        play_data_aug = get_equi_data(play_data, config.board_width,
                                      config.board_height)
        rpm.extend(play_data_aug)

        print('Eposide i:{}, episode_len:{}'.format(i + 1, len(play_data)))

        if len(rpm) > config.batch_size:
            loss, entropy = policy_update(agent=alphazero,
                                          buffer=rpm,
                                          config=config)
        # check the performance of the current model,
        # and save the model params
        if (i + 1) % config.check_freq == 0:
            print('current config-play batch: {}'.format(i + 1))
            win_ratio = policy_evaluate(agent=alphazero,
                                        game=gomokugame,
                                        config=config)
            alphazero.save(save_dir=config.work_dir)
            if win_ratio > config.best_win_ratio:
                print('New best policy!!!!!!!!')
                config.best_win_ratio = win_ratio
                # update the best_policy
                alphazero.save(save_dir=config.work_dir)
                if (config.best_win_ratio == 1.0
                        and config.pure_mcts_playout_num < 5000):
                    config.pure_mcts_playout_num += 1000
                    config.best_win_ratio = 0.0


if __name__ == '__main__':
    main()
