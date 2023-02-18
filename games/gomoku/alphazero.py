import datetime
import itertools
import json
import os
import sys
import tempfile
import time
import traceback
import functools
import numpy as np
from absl import app
from agent import AlphaZeroAgent
from config import Config
from evaluator import AlphaZeroEvaluator
from game_openspiel import GomokuGame
from open_spiel.python.utils import data_logger, spawn, stats, file_logger
from replaybuffer import Buffer, TrainInput, Trajectory, TrajectoryState
import sys

sys.path.append('../../')
from muzero.mcts.mcts_deepmind import (MCTSBot, RandomRolloutEvaluator,
                                       SearchNode)


def watcher(fn):
    """A decorator to fn/processes that gives a logger and logs exceptions."""
    @functools.wraps(fn)
    def _watcher(*, config, num=None, **kwargs):
        """Wrap the decorated function."""
        name = fn.__name__
        if num is not None:
            name += '-' + str(num)
        with file_logger.FileLogger(config.path, name, config.quiet) as logger:
            print('{} started'.format(name))
            logger.print('{} started'.format(name))
            try:
                return fn(config=config, logger=logger, **kwargs)
            except Exception as e:
                logger.print('\n'.join([
                    '',
                    ' Exception caught '.center(60, '='),
                    traceback.format_exc(),
                    '=' * 60,
                ]))
                print('Exception caught in {}: {}'.format(name, e))
                raise
            finally:
                logger.print('{} exiting'.format(name))
                print('{} exiting'.format(name))

    return _watcher


def init_bot(config: Config, game, evaluator_, evaluation):
    """Initializes a bot."""
    noise = None if evaluation else (config.policy_epsilon,
                                     config.policy_alpha)
    return MCTSBot(game,
                   config.uct_c,
                   config.max_simulations,
                   evaluator_,
                   solve=False,
                   dirichlet_noise=noise,
                   child_selection_fn=SearchNode.puct_value,
                   verbose=False,
                   dont_return_chance_node=True)


def play_game(logger, game_num, game, bots, temperature, temperature_drop):
    """Play one game, return the trajectory."""
    trajectory = Trajectory()
    actions = []
    state = game.new_initial_state()
    random_state = np.random.RandomState()
    logger.opt_print(' Starting game {} '.format(game_num).center(60, '-'))
    logger.opt_print('Initial state:\n{}'.format(state))
    while not state.is_terminal():
        if state.is_chance_node():
            # For chance nodes, rollout according to chance node's probability
            # distribution
            outcomes = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes)
            action = random_state.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            root = bots[state.current_player()].mcts_search(state)
            policy = np.zeros(game.num_distinct_actions())
            for c in root.children:
                policy[c.action] = c.explore_count
            policy = policy**(1 / temperature)
            policy /= policy.sum()
            if len(actions) >= temperature_drop:
                action = root.best_child().action
            else:
                action = np.random.choice(len(policy), p=policy)
            trajectory.states.append(
                TrajectoryState(state.observation_tensor(),
                                state.current_player(),
                                state.legal_actions_mask(), action, policy,
                                root.total_reward / root.explore_count))
            action_str = state.action_to_string(state.current_player(), action)
            actions.append(action_str)
            logger.opt_print('Player {} sampled action: {}'.format(
                state.current_player(), action_str))
            state.apply_action(action)
    logger.opt_print('Next state:\n{}'.format(state))

    trajectory.returns = state.returns()
    logger.print('Game {}: Returns: {}; Actions: {}'.format(
        game_num, ' '.join(map(str, trajectory.returns)), ' '.join(actions)))
    return trajectory

@watcher
def actor(*, config: Config, game, logger, queue):
    """An actor process runner that generates games and returns
    trajectories."""
    logger.print('Initializing model')
    alphazero_agent = AlphaZeroAgent(config.num_rows, config.num_cols,
                                     config.learning_rate, config.weight_decay)
    logger.print('Initializing bots')
    az_evaluator = AlphaZeroEvaluator(game, alphazero_agent)
    bots = [
        init_bot(config, game, az_evaluator, False),
        init_bot(config, game, az_evaluator, False),
    ]
    for game_num in itertools.count():

        queue.put(
            play_game(logger, game_num, game, bots, config.temperature,
                      config.temperature_drop))

@watcher
def evaluator(*, game, config: Config, logger, queue):
    """A process that plays the latest checkpoint vs standard MCTS."""
    results = Buffer(config.evaluation_window)
    logger.print('Initializing model')
    alphazero_agent = AlphaZeroAgent(config.num_rows, config.num_cols,
                                     config.learning_rate, config.weight_decay)
    logger.print('Initializing bots')
    az_evaluator = AlphaZeroEvaluator(game, alphazero_agent)
    random_evaluator = RandomRolloutEvaluator()

    for game_num in itertools.count():
        az_player = game_num % 2
        difficulty = (game_num // 2) % config.eval_levels
        max_simulations = int(config.max_simulations * (10**(difficulty / 2)))
        bots = [
            init_bot(config, game, az_evaluator, True),
            MCTSBot(game,
                    config.uct_c,
                    max_simulations,
                    random_evaluator,
                    solve=True,
                    verbose=False,
                    dont_return_chance_node=True)
        ]
        if az_player == 1:
            bots = list(reversed(bots))

        trajectory = play_game(logger,
                               game_num,
                               game,
                               bots,
                               temperature=1,
                               temperature_drop=0)
        results.append(trajectory.returns[az_player])
        queue.put((difficulty, trajectory.returns[az_player]))

        logger.print('AZ: {}, MCTS: {}, AZ avg/{}: {:.3f}'.format(
            trajectory.returns[az_player], trajectory.returns[1 - az_player],
            len(results), np.mean(results.data)))

@watcher
def learner(*, game, config: Config, actors, evaluators, broadcast_fn, logger):
    """A learner that consumes the replay buffer and trains the network."""
    logger.also_to_stdout = True

    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    logger.print('Initializing model')
    alphazero_agent = AlphaZeroAgent(config.num_rows, config.num_cols,
                                     config.learning_rate, config.weight_decay)
    data_log = data_logger.DataLoggerJsonLines(config.path, 'learner', True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(['Player1', 'Player2', 'Draw'])
    evals = [
        Buffer(config.evaluation_window) for _ in range(config.eval_levels)
    ]
    total_trajectories = 0

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_trajectories():
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            replay_buffer.extend(
                TrainInput(s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in trajectory.states)

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count -
                                                                 1)
                n = trajectory.states[index]
                accurate = (n.value >=
                            0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a
        checkpoint."""
        losses = []
        for _ in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            batch = TrainInput.stack(data)
            feed_dict = {
                'observation': batch.observation,
                'legals_mask': batch.legals_mask,
                'policy_targets': batch.policy,
                'value_targets': batch.value
            }
            loss, policy_loss, value_loss, entroy = alphazero_agent.learn(
                feed_dict)
            losses.append(loss)
        return losses

    last_time = time.time() - 60
    for step in itertools.count(1):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now

        logger.print('Step:', step)
        logger.print(
            ('Collected {:5} states from {:3} games, {:.1f} states/s. '
             '{:.1f} states/(s*actor), game length: {:.1f}').format(
                 num_states, num_trajectories, num_states / seconds,
                 num_states / (config.actors * seconds),
                 num_states / num_trajectories))
        logger.print('Buffer size: {}. States seen: {}'.format(
            len(replay_buffer), replay_buffer.total_seen))

        losses = learn(step)

        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write({
            'step': step,
            'total_states': replay_buffer.total_seen,
            'states_per_s': num_states / seconds,
            'states_per_s_actor': num_states / (config.actors * seconds),
            'total_trajectories': total_trajectories,
            'trajectories_per_s': num_trajectories / seconds,
            'queue_size': 0,  # Only available in C++.
            'game_length': game_lengths.as_dict,
            'game_length_hist': game_lengths_hist.data,
            'outcomes': outcomes.data,
            'value_accuracy': [v.as_dict for v in value_accuracies],
            'value_prediction': [v.as_dict for v in value_predictions],
            'eval': {
                'count': evals[0].total_seen,
                'results': [sum(e.data) / len(e) if e else 0 for e in evals],
            },
            'batch_size': batch_size_stats.as_dict,
            'batch_size_hist': [0, 1],
            'loss': {
                'sum': losses,
            },
            'cache': {  # Null stats because it's hard to report between processes.
                'size': 0,
                'max_size': 0,
                'usage': 0,
                'requests': 0,
                'requests_per_s': 0,
                'hits': 0,
                'misses': 0,
                'misses_per_s': 0,
                'hit_rate': 0,
            },
        })
        logger.print()

        if config.max_steps > 0 and step >= config.max_steps:
            break


def alpha_zero(config: Config):
    """Start all the worker processes for a full alphazero setup."""
    game = GomokuGame()
    config.observation_shape = game.observation_tensor_shape()
    config.output_size = game.num_distinct_actions()

    print('Starting game', config.game)
    if game.num_players() != 2:
        sys.exit('AlphaZero can only handle 2-player games.')
    path = config.path
    if not path:
        path = tempfile.mkdtemp(prefix='az-{}-{}-'.format(
            datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'), config.game))
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))

    actors = [
        spawn.Process(actor, kwargs={
            'game': game,
            'config': config,
            'num': i
        }) for i in range(config.actors)
    ]
    evaluators = [
        spawn.Process(evaluator,
                      kwargs={
                          'game': game,
                          'config': config,
                          'num': i
                      }) for i in range(config.evaluators)
    ]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
        learner(
            game=game,
            config=config,
            actors=actors,  # pylint: disable=missing-kwoa
            evaluators=evaluators,
            broadcast_fn=broadcast)
    except (KeyboardInterrupt, EOFError):
        print('Caught a KeyboardInterrupt, stopping early.')
    finally:
        broadcast('')
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(0.001)
        for proc in evaluators:
            proc.join()


def main(unused_argv):
    config = Config()
    alpha_zero(config)


if __name__ == '__main__':
    with spawn.main_handler():
        app.run(main)
