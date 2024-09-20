import os
import pprint
import threading
import timeit
import traceback
from collections import deque
from queue import Queue
from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pettingzoo import AECEnv
from rlcard.utils import run_game_pettingzoo
from torch import multiprocessing as mp
from torch.optim import Optimizer

from rlzero.agents.rl_args import RLArguments
from rlzero.models import DMCAgent, DMCModel, DMCModelPettingZoo
from rlzero.utils.logger_utils import get_logger

logger = get_logger('rlzero')


class DMCTrainerPettingzoo(object):
    """A distributed multi-agent reinforcement learning system for training
    DouDizhu game AI agents.

    This class handles:
    - Creating buffers to store experience data.
    - Using multi-threading for batch processing and learning.
    - Periodically saving checkpoints.
    - Logging and outputting training statistics.
    """

    def __init__(
        self,
        env: AECEnv,
        is_pettingzoo_env: bool = False,
        args: RLArguments = RLArguments,
    ) -> None:
        """Initialize the DistributedDouZero system.

        Args:
            args: Configuration arguments for the training process.
        """
        self.env = env
        self.is_pettingzoo_env = is_pettingzoo_env
        if not self.is_pettingzoo_env:
            self.num_players = env.num_players
            self.action_shape = env.action_shape
            if self.action_shape[0] is None:  # One-hot encoding
                self.action_shape = [[self.env.num_actions]
                                     for _ in range(self.num_players)]

            def model_func(device) -> DMCModel:
                return DMCModel(
                    self.env.state_shape,
                    self.action_shape,
                    exp_epsilon=self.args.epsilon_greedy,
                    device=str(device),
                )
        else:
            self.num_players = self.env.num_agents

            def model_func(device) -> DMCModelPettingZoo:
                return DMCModelPettingZoo(self.env,
                                          exp_epsilon=self.args.epsilon_greedy,
                                          device=device)

        self.model_func = model_func
        self.mean_episode_return_buf = {
            p: deque(maxlen=100)
            for p in range(self.num_players)
        }

        self.args: RLArguments = args
        self.check_and_init_device()
        # Initialize actor models
        self.init_actor_models()

        # Initialize learner model
        self.learner_model = self.model_func(self.args.training_device)

        # Initialize buffers
        self.buffers = self.create_buffers(self.device_iterator)
        # Initialize Optimizers
        self.optimizers = self.create_optimizers(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            epsilon=self.args.epsilon,
            alpha=self.args.alpha,
        )

        # Initialize queues
        self.init_queues()

        self.checkpoint_path = os.path.expandvars(
            os.path.expanduser(
                f'{self.args.savedir}/{self.args.project}/model.tar'))

        self.stata_info_keys = []
        for p in range(self.num_players):
            self.stata_info_keys.append('mean_episode_return_' + str(p))
            self.stata_info_keys.append('loss_' + str(p))

        # Initialize global step and stat info
        self.global_step = 0
        self.stata_info = {k: 0 for k in self.stata_info_keys}
        self.global_player_step = {
            player_id: 0
            for player_id in self.player_ids
        }

    def init_actor_models(self) -> None:
        """Initialize actor models for each device."""
        self.actor_models = {}
        for device in self.device_iterator:
            model = self.model_func(device=device)
            model.share_memory()
            model.eval()
            self.actor_models[device] = model

    def check_and_init_device(self) -> None:
        """Check and initialize the device."""
        if not self.args.actor_device_cpu or self.args.training_device != 'cpu':
            if not torch.cuda.is_available():
                raise AssertionError(
                    'CUDA not available. If you have GPUs, specify their IDs with --gpu_devices. '
                    'Otherwise, train with CPU using: python3 train.py --actor_device_cpu --training_device cpu'
                )
        if self.args.actor_device_cpu:
            self.device_iterator = ['cpu']
        else:
            self.device_iterator = range(self.args.num_actor_devices)
            assert (
                self.args.num_actor_devices <= len(
                    self.args.gpu_devices.split(','))
            ), 'The number of actor devices can not exceed the number of available devices'

    def init_queues(self) -> None:
        """Initialize multiprocessing queues for communication between
        processes."""
        ctx = mp.get_context('spawn')
        self.free_queue = {}
        self.full_queue = {}

        for device in self.device_iterator:
            self.free_queue[device] = {
                key: ctx.SimpleQueue()
                for key in range(self.num_players)
            }
            self.full_queue[device] = {
                key: ctx.SimpleQueue()
                for key in range(self.num_players)
            }

    def create_optimizers(
        self,
        learning_rate: float,
        momentum: float,
        epsilon: float,
        alpha: float,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for each player model.

        Args:
            learning_rate: Learning rate for the optimizer.
            momentum: Momentum for the optimizer.
            epsilon: Epsilon for the optimizer.
            alpha: Alpha for the optimizer.

        Returns:
            Dictionary of optimizers for each player model.
        """
        optimizers = {}

        for player_id in range(self.num_players):
            model_parameters = self.learner_model.parameters(player_id)
            optimizer = torch.optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                momentum=momentum,
                eps=epsilon,
                alpha=alpha,
            )
            optimizers[player_id] = optimizer

        return optimizers

    def create_buffers(
        self,
        rollout_length: int,
        num_buffers: int,
        state_shape: Tuple[int],
        action_shape: Tuple[int],
        device_iterator: Iterator[int],
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Create buffers for each player and device.

        Args:
            rollout_length: The length of the rollout.
            num_buffers: The number of buffers.
            state_shape: The shape of the state.
            action_shape: The shape of the action.
            device_iterator: Iterable of device indices (GPU or CPU).

        Returns:
            Dictionary of buffers for each device and player.
        """
        buffers = {}
        for device in device_iterator:
            buffers[device] = {}
            for player_id in range(self.num_players):

                specs = {
                    'done':
                    dict(size=(rollout_length, ), dtype=torch.bool),
                    'episode_return':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'target':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'state':
                    dict(size=(rollout_length, ) +
                         tuple(state_shape[player_id]),
                         dtype=torch.int8),
                    'action':
                    dict(size=(rollout_length, ) +
                         tuple(action_shape[player_id]),
                         dtype=torch.int8),
                }

                player_buffers = {key: [] for key in specs}

                # Create buffers for each player
                for _ in range(num_buffers):
                    for key, spec in specs.items():
                        if device != 'cpu':
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device(
                                    f'cuda:{device}')).share_memory_())
                        else:
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device('cpu')).share_memory_())

                        player_buffers[key].append(buffer_tensor)

                buffers[device][player_id] = player_buffers

        return buffers

    def create_buffers_pettingzoo(
        self, rollout_length: int, num_buffers: int,
        device_iterator: Iterator[int]
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Create buffers for each player and device.

        Args:
            rollout_length: The length of the rollout.
            num_buffers: The number of buffers.
            state_shape: The shape of the state.
            action_shape: The shape of the action.
            device_iterator: Iterable of device indices (GPU or CPU).

        Returns:
            Dictionary of buffers for each device and player.
        """
        buffers = {}
        for device in device_iterator:
            buffers[device] = {}
            for agent_name in self.env.agents:
                state_shape = self.env.observation_space(
                    agent_name)['observation'].shape
                action_shape = self.env.action_space(agent_name).n

                specs = {
                    'done':
                    dict(size=(rollout_length, ), dtype=torch.bool),
                    'episode_return':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'target':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'state':
                    dict(size=(rollout_length, ) + tuple(state_shape),
                         dtype=torch.int8),
                    'action':
                    dict(size=(rollout_length, ) + tuple(action_shape),
                         dtype=torch.int8),
                }

                player_buffers = {key: [] for key in specs}

                # Create buffers for each player
                for _ in range(num_buffers):
                    for key, spec in specs.items():
                        if device != 'cpu':
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device(
                                    f'cuda:{device}')).share_memory_())
                        else:
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device('cpu')).share_memory_())

                        player_buffers[key].append(buffer_tensor)

                buffers[device][agent_name] = player_buffers

        return buffers

    def get_batch(
        self,
        free_queue: Queue,
        full_queue: Queue,
        buffers: Dict[str, torch.Tensor],
        lock: threading.Lock,
    ) -> Dict[str, torch.Tensor]:
        """Samples a batch from the `buffers` using indices retrieved from
        `full_queue`. After sampling, it frees the indices by sending them to
        `free_queue`.

        Args:
            free_queue (Queue): A queue where free buffer indices are placed after being processed.
            full_queue (Queue): A queue from which buffer indices are retrieved for batch sampling.
            buffers (Dict[str, torch.Tensor]): A dictionary of tensors containing the data to be batched.
            lock (Lock): A threading lock to ensure thread-safe access to shared resources.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the sampled batch, with the same keys as `buffers`.
        """

        # Thread-safe section using the provided lock
        with lock:
            # Retrieve a batch of indices from the full_queue
            indices = [full_queue.get() for _ in range(self.args.batch_size)]

        # Create a batch by stacking the selected elements from each buffer
        batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1)
            for key in buffers
        }

        # Release the indices back to the free_queue for future use
        for m in indices:
            free_queue.put(m)

        return batch

    def get_action(
        self,
        worker_id: int,
        free_queue: Dict[str, mp.Queue],
        full_queue: Dict[str, mp.Queue],
        model: DMCModel,
        buffers: Dict[str, Dict[str, List[torch.Tensor]]],
        device: Union[str, int],
    ) -> None:
        """Actor process that interacts with the environment and fills buffers
        with data.

        Args:
            worker_id: Process index for the actor.
            free_queue: Queue for getting free buffer indices.
            full_queue: Queue for passing filled buffer indices to the main process.
            model: The model used to get actions from the environment.
            buffers: Buffers to store experiences.
            device: Device name ('cpu' or 'cuda:x') where this actor will run.
        """
        rollout_length = self.args.rollout_length
        try:
            logger.info('Device %s Actor %i started.', str(device), worker_id)
            self.env.seed(worker_id)
            self.env.set_agents(model.get_agents())

            done_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.num_players)
            }
            episode_return_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.num_players)
            }
            target_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.num_players)
            }
            state_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.num_players)
            }
            action_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.num_players)
            }
            size = {p: 0 for p in range(self.num_players)}

            while True:
                trajectories, payoffs = self.env.run(is_training=True)
                for p in range(self.num_players):
                    size[p] += len(trajectories[p][:-1]) // 2
                    diff = size[p] - len(target_buf[p])
                    if diff > 0:
                        done_buf[p].extend([False for _ in range(diff - 1)])
                        done_buf[p].append(True)
                        episode_return_buf[p].extend(
                            [0.0 for _ in range(diff - 1)])
                        episode_return_buf[p].append(float(payoffs[p]))
                        target_buf[p].extend(
                            [float(payoffs[p]) for _ in range(diff)])
                        # State and action
                        for i in range(0, len(trajectories[p]) - 2, 2):
                            state = trajectories[p][i]['obs']
                            action = self.env.get_action_feature(
                                trajectories[p][i + 1])
                            state_buf[p].append(torch.from_numpy(state))
                            action_buf[p].append(torch.from_numpy(action))

                    while size[p] > rollout_length:
                        index = free_queue[p].get()
                        if index is None:
                            break
                        for t in range(rollout_length):
                            buffers[p]['done'][index][t, ...] = done_buf[p][t]
                            buffers[p]['episode_return'][index][
                                t, ...] = episode_return_buf[p][t]
                            buffers[p]['target'][index][t,
                                                        ...] = target_buf[p][t]
                            buffers[p]['state'][index][t,
                                                       ...] = state_buf[p][t]
                            buffers[p]['action'][index][t,
                                                        ...] = action_buf[p][t]
                        full_queue[p].put(index)
                        done_buf[p] = done_buf[p][rollout_length:]
                        episode_return_buf[p] = episode_return_buf[p][
                            rollout_length:]
                        target_buf[p] = target_buf[p][rollout_length:]
                        state_buf[p] = state_buf[p][rollout_length:]
                        action_buf[p] = action_buf[p][rollout_length:]
                        size[p] -= rollout_length

        except KeyboardInterrupt:
            logger.info('Actor %i stopped manually.', worker_id)
        except Exception as e:
            logger.error(f'Exception in worker process {worker_id}: {str(e)}')
            traceback.print_exc()
            raise e

    def get_action_pettingzoo(
        self,
        worker_id: int,
        free_queue: Dict[str, mp.Queue],
        full_queue: Dict[str, mp.Queue],
        model: DMCModelPettingZoo,
        buffers: Dict[str, Dict[str, List[torch.Tensor]]],
        device: Union[str, int],
    ) -> None:
        """Actor process that interacts with the environment and fills buffers
        with data.

        Args:
            worker_id: Process index for the actor.
            free_queue: Queue for getting free buffer indices.
            full_queue: Queue for passing filled buffer indices to the main process.
            model: The model used to get actions from the environment.
            buffers: Buffers to store experiences.
            device: Device name ('cpu' or 'cuda:x') where this actor will run.
        """
        rollout_length = self.args.rollout_length
        try:
            logger.info('Device %s Actor %i started.', str(device), worker_id)
            done_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.env.num_agents)
            }
            episode_return_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.env.num_agents)
            }
            target_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.env.num_agents)
            }
            state_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.env.num_agents)
            }
            action_buf = {
                p: deque(maxlen=rollout_length)
                for p in range(self.env.num_agents)
            }
            size = {p: 0 for p in range(self.env.num_agents)}

            while True:
                trajectories = run_game_pettingzoo(self.env,
                                                   model.agents,
                                                   is_training=True)
                for agent_id, agent_name in enumerate(
                        self.env.possible_agents):
                    traj_size = len(trajectories[agent_name]) // 2
                    if traj_size > 0:
                        size[agent_id] += traj_size
                        target_return = trajectories[agent_name][-2][1]
                        target_buf[agent_id].extend([target_return] *
                                                    traj_size)
                        for i in range(0, len(trajectories[agent_name]), 2):
                            state = trajectories[agent_name][i][0][
                                'observation']
                            action = self._get_action_feature(
                                trajectories[agent_name][i + 1],
                                model.agents[agent_name].action_shape)
                            episode_return = trajectories[agent_name][i][1]
                            done = trajectories[agent_name][i][2]

                            state_buf[agent_id].append(torch.from_numpy(state))
                            action_buf[agent_id].append(
                                torch.from_numpy(action))
                            episode_return_buf[agent_id].append(episode_return)
                            done_buf[agent_id].append(done)

                while size[agent_id] > rollout_length:
                    index = free_queue[agent_id].get()
                    if index is None:
                        break

                    for t in range(rollout_length):
                        temp_done = done_buf[agent_id][t]
                        buffers[agent_id]['done'][index][t, ...] = temp_done
                        buffers[agent_id]['episode_return'][index][
                            t, ...] = episode_return_buf[agent_id][t]
                        buffers[agent_id]['target'][index][
                            t, ...] = target_buf[agent_id][t]
                        buffers[agent_id]['state'][index][
                            t, ...] = state_buf[agent_id][t]
                        buffers[agent_id]['action'][index][
                            t, ...] = action_buf[agent_id][t]

                    full_queue[agent_id].put(index)

                    done_buf[agent_id] = done_buf[agent_id][rollout_length:]
                    episode_return_buf[agent_id] = episode_return_buf[
                        agent_id][rollout_length:]
                    target_buf[agent_id] = target_buf[agent_id][
                        rollout_length:]
                    state_buf[agent_id] = state_buf[agent_id][rollout_length:]
                    action_buf[agent_id] = action_buf[agent_id][
                        rollout_length:]
                    size[agent_id] -= rollout_length

        except KeyboardInterrupt:
            logger.info('Actor %i stopped manually.', worker_id)
        except Exception as e:
            logger.error(f'Exception in worker process {worker_id}: {str(e)}')
            traceback.print_exc()
            raise e

    def _get_action_feature(self, action: int, action_space: int):
        out = np.zeros(action_space)
        out[action] = 1
        return out

    def learn(
        self,
        agent: DMCAgent,
        optimizer: Optimizer,
        player_id: str,
        batch: Dict[str, torch.Tensor],
        lock: threading.Lock,
    ) -> Dict[str, Union[float, int]]:
        """Perform a single learning (optimization) step for a given player.

        Args:
            model: The learner's model to be optimized.
            optimizer: The optimizer used to update the model.
            player: The player in the game ('landlord', 'landlord_up', or 'landlord_down').
            batch: A batch of experiences from the environment.
            lock: Lock object to synchronize updates across threads.

        Returns:
            Dictionary with statistics about the learning step.
        """
        device = torch.device(f'cuda:{self.args.training_device}'
                              if self.args.training_device != 'cpu' else 'cpu')

        state = torch.flatten(batch['state'].to(device), 0, 1).float()
        action = torch.flatten(batch['action'].to(device), 0, 1).float()
        target = torch.flatten(batch['target'].to(device), 0, 1)
        episode_returns = batch['episode_return'][batch['done']]

        self.mean_episode_return_buf[player_id].append(
            torch.mean(episode_returns).to(device))

        with lock:
            values = agent.forward(state, action)
            loss = self.compute_loss(values, target)

            stats = {
                f'mean_episode_return_{player_id}':
                torch.mean(
                    torch.stack([
                        _r for _r in self.mean_episode_return_buf[player_id]
                    ])).item(),
                f'loss_{player_id}':
                loss.item(),
            }

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(),
                                     self.args.max_grad_norm)
            optimizer.step()

            for actor_model in self.actor_models.values():
                actor_model.get_agent(player_id).load_state_dict(
                    agent.state_dict())

        return stats

    def compute_loss(self, pred_values: torch.Tensor,
                     target: torch.Tensor) -> torch.Tensor:
        """Compute the loss between predicted values and target values.

        Args:
            pred_values: Predicted values from the model.
            target: Ground truth target values.

        Returns:
            Loss tensor.
        """
        return F.mse_loss(pred_values, target, reduction='mean')

    def batch_and_learn(
        self,
        thread_id: int,
        player_id: str,
        local_lock: threading.Lock,
        player_lock: threading.Lock,
        lock: threading.Lock,
        device: Union[str, int],
    ) -> None:
        """Thread function that batches data and performs learning.

        Args:
            thread_id: Thread index.
            player_id: Player identifier.
            local_lock: Lock for local thread synchronization.
            player_lock: Lock for player-specific synchronization.
            lock: Global lock for synchronization.
            device: Device identifier.
        """
        while self.global_step < self.args.total_steps:
            batch_data = self.get_batch(
                self.free_queue[device][player_id],
                self.full_queue[device][player_id],
                self.buffers[device][player_id],
                local_lock,
            )
            learner_stats = self.learn(
                self.learner_model.get_agent(player_id),
                self.optimizers[player_id],
                player_id,
                batch_data,
                player_lock,
            )

            with lock:
                for key in learner_stats:
                    self.stata_info[key] = learner_stats[key]
                self.global_step += self.args.rollout_length * self.args.batch_size
                self.global_player_step[player_id] += (
                    self.args.rollout_length * self.args.batch_size)

    def train(self) -> None:
        """Main function for training.

        Initializes necessary components such as buffers, models, optimizers,
        and actors, then spawns subprocesses for actors and threads for
        learning. It also handles logging and checkpointing.
        """

        # Initialize the actor processes
        ctx = mp.get_context('spawn')
        # Start actor processes
        actor_processes = []
        for device in self.device_iterator:
            for worker_id in range(self.args.num_actors):
                actor_process = ctx.Process(
                    target=self.get_action,
                    args=(
                        worker_id,
                        self.free_queue[device],
                        self.full_queue[device],
                        self.actor_models[device],
                        self.buffers[device],
                        device,
                    ),
                )
                actor_process.start()
                actor_processes.append(actor_process)

        # Initialize free_queue
        for device in self.device_iterator:
            for m in range(self.args.num_buffers):
                for player_id in range(self.num_players):
                    self.free_queue[device][player_id].put(m)

        # Initialize threads, locks, player_locks
        threads = []
        locks = {
            device: {
                player_id: threading.Lock()
                for player_id in range(self.num_players)
            }
            for device in self.device_iterator
        }
        player_locks = [threading.Lock() for _ in range(self.num_players)]

        # Start learning threads
        for device in self.device_iterator:
            for i in range(self.args.num_threads):
                for player_id in range(self.num_players):
                    thread = threading.Thread(
                        target=self.batch_and_learn,
                        name=f'batch-and-learn-{i}',
                        args=(
                            i,
                            player_id,
                            locks[device][player_id],
                            player_locks[player_id],
                            device,
                        ),
                    )
                    thread.start()
                    threads.append(thread)

        fps_log = deque(maxlen=24)
        timer = timeit.default_timer

        try:
            # Main training loop
            last_checkpoint_time = timer() - self.args.save_interval * 60
            while self.global_step < self.args.total_steps:
                current_step, current_player_step = (
                    self.global_step,
                    self.global_player_step.copy(),
                )
                start_time = timer()

                if timer(
                ) - last_checkpoint_time > self.args.save_interval * 60:
                    self.save_checkpoint(self.checkpoint_path)
                    last_checkpoint_time = timer()

                end_time = timer()
                fps = (self.global_step - current_step) / (end_time -
                                                           start_time)

                fps_log.append(fps)
                fps_avg = np.mean(fps_log)

                player_fps = {
                    player_id:
                    (self.global_player_step[player_id] -
                     current_player_step[player_id]) / (end_time - start_time)
                    for player_id in range(self.num_players)
                }
                if self.global_step % 1000 == 0:
                    logger.info(
                        'After %i steps: @ %.1f fps (avg@ %.1f fps) Stats:\n%s',
                        self.global_step,
                        fps,
                        fps_avg,
                        pprint.pformat(self.stata_info),
                    )
                    for player_id in range(self.num_players):
                        logger.info(
                            'Player %i: @ %.1f fps (avg@ %.1f fps)',
                            player_id,
                            player_fps[player_id],
                            fps_avg,
                        )

        except KeyboardInterrupt:
            pass
        finally:
            # End learning threads
            for thread in threads:
                thread.join()
            logger.info('Training finished after %d steps.', self.global_step)
            self.save_checkpoint(self.checkpoint_path)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoints.

        Args:
            checkpoint_path: Path to save the checkpoint.
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        if self.args.disable_checkpoint:
            return
        logger.info('Saving checkpoint to %s', checkpoint_path)
        torch.save(
            {
                'model_state_dict': {
                    player_id:
                    self.learner_model.get_agent(player_id).state_dict()
                    for player_id in range(self.num_players)
                },
                'optimizer_state_dict': {
                    player_id: self.optimizers[player_id].state_dict()
                    for player_id in range(self.num_players)
                },
                'stata_info': self.stata_info,
                'global_step': self.global_step,
                'global_player_step': self.global_player_step,
            },
            checkpoint_path,
        )
        for player_id in range(self.num_players):
            model_weights_dir = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (
                    self.args.savedir,
                    self.args.project,
                    str(player_id) + '_weights_' + str(self.global_step) +
                    '.pth',
                )))
            torch.save(
                self.learner_model.get_agent(player_id).state_dict(),
                model_weights_dir)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoints.

        Args:
            checkpoint_path: Path to load the checkpoint.
        """
        if self.args.load_model and os.path.exists(checkpoint_path):
            checkpoint_states = torch.load(
                checkpoint_path,
                map_location=(f'cuda:{self.args.training_device}' if
                              self.args.training_device != 'cpu' else 'cpu'),
            )
            for player_id in range(self.num_players):
                self.learner_model.get_agent(player_id).load_state_dict(
                    checkpoint_states['model_state_dict'][player_id])
                self.optimizers[player_id].load_state_dict(
                    checkpoint_states['optimizer_state_dict'][player_id])
                for device in self.device_iterator:
                    self.actor_models[device].get_agent(
                        player_id).load_state_dict(
                            self.learner_model.get_agent(
                                player_id).state_dict())
            self.stata_info = checkpoint_states['stata_info']
            self.global_step = checkpoint_states['global_step']
            self.global_player_step = checkpoint_states['global_player_step']
            logger.info(
                f'Resuming preempted job, current stats:\n{self.stata_info}')
