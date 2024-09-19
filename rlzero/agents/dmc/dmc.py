import multiprocessing as mp
import os
import pprint
import threading
import time
import timeit
import traceback
from collections import deque
from queue import Queue
from threading import Lock
from typing import Dict, Iterator, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from rlzero.agents.dmc.env_utils import EnvWrapper
from rlzero.agents.dmc.utils import cards2tensor
from rlzero.agents.rl_args import RLArguments
from rlzero.envs.doudizhu.env import DouDiZhuEnv
from rlzero.models.doudizhu import DouDiZhuModel
from rlzero.utils.logger_utils import get_logger

logger = get_logger(__name__)


class DMCAgent:
    """A distributed multi-agent reinforcement learning system for training
    DouDizhu game AI agents.

    This class handles:
    - Creating buffers to store experience data.
    - Using multi-threading for batch processing and learning.
    - Periodically saving checkpoints.
    - Logging and outputting training statistics.
    """

    def __init__(self, args: RLArguments = RLArguments) -> None:
        self.positions: List[str] = [
            'landlord', 'landlord_up', 'landlord_down'
        ]
        self.mean_episode_return_buf = {
            p: deque(maxlen=100)
            for p in self.positions
        }
        self.args: RLArguments = args
        self.checkpoint_path = os.path.expandvars(
            os.path.expanduser(
                f'{self.args.savedir}/{self.args.project}/model.tar'))
        self.stata_info_keys = [
            'loss_landlord',
            'loss_landlord_up',
            'loss_landlord_down',
            'mean_episode_return_landlord',
            'mean_episode_return_landlord_up',
            'mean_episode_return_landlord_down',
        ]

        self.check_and_init_device()

        # Initialize actor models
        self.init_actor_models()

        # Initialize learner model
        self.learner_model = DouDiZhuModel(device=self.args.training_device)

        # Initialize buffers
        self.buffers = self.create_buffers(
            num_buffers=self.args.num_buffers,
            rollout_length=self.args.rollout_length,
            device_iterator=self.device_iterator,
        )
        # Initialize Optimizers
        self.optimizers = self.create_optimizers(
            learning_rate=self.args.learning_rate,
            momentum=self.args.momentum,
            epsilon=self.args.epsilon,
            alpha=self.args.alpha,
        )

    def init_actor_models(self) -> None:
        """Initialize actor models."""
        self.actor_models = {}
        for device in self.device_iterator:
            model = DouDiZhuModel(device=device)
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

    # Initialize queues
    def init_queues(self) -> None:
        ctx = mp.get_context('spawn')
        self.free_queue = {}
        self.full_queue = {}

        for device in self.device_iterator:
            self.free_queue[device] = {
                'landlord': ctx.SimpleQueue(),
                'landlord_up': ctx.SimpleQueue(),
                'landlord_down': ctx.SimpleQueue(),
            }
            self.full_queue[device] = {
                'landlord': ctx.SimpleQueue(),
                'landlord_up': ctx.SimpleQueue(),
                'landlord_down': ctx.SimpleQueue(),
            }

    def create_optimizers(
        self,
        learning_rate: float,
        momentum: float,
        epsilon: float,
        alpha: float,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for each position model.

        Args:
            learning_rate: Learning rate for the optimizer.
            momentum: Momentum for the optimizer.
            epsilon: Epsilon for the optimizer.
            alpha: Alpha for the optimizer.

        Returns:
            Dictionary of optimizers for each position model.
        """
        optimizers = {}

        for position in self.positions:
            model_parameters = getattr(self.learner_model,
                                       position).parameters()
            optimizer = torch.optim.RMSprop(
                model_parameters,
                lr=learning_rate,
                momentum=momentum,
                eps=epsilon,
                alpha=alpha,
            )
            optimizers[position] = optimizer

        return optimizers

    def get_batch(
        self,
        free_queue: Queue,
        full_queue: Queue,
        buffers: Dict[str, torch.Tensor],
        lock: Lock,
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

    def create_buffers(
        self,
        num_buffers: int,
        rollout_length: int,
        device_iterator: Iterator[int],
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Create buffers for each position and device.

        Args:
            num_buffers: Number of buffers to create.
            rollout_length: Length of the rollout.
            device_iterator: Iterable of device indices (GPU or CPU).

        Returns:
            Dictionary of buffers for each device and position.
        """
        buffers = {}

        for device in device_iterator:
            buffers[device] = {}

            for position in self.positions:
                feature_dim = 319 if position == 'landlord' else 430

                specs = {
                    'done':
                    dict(size=(rollout_length, ), dtype=torch.bool),
                    'episode_return':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'target':
                    dict(size=(rollout_length, ), dtype=torch.float32),
                    'obs_x_no_action':
                    dict(size=(rollout_length, feature_dim), dtype=torch.int8),
                    'obs_action':
                    dict(size=(rollout_length, 54), dtype=torch.int8),
                    'obs_z':
                    dict(size=(rollout_length, 5, 162), dtype=torch.int8),
                }

                position_buffers: Dict[str, List[torch.Tensor]] = {
                    key: []
                    for key in specs
                }

                for _ in range(num_buffers):
                    for key, spec in specs.items():
                        if device != 'cpu':
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device(
                                    f'cuda:{device}')).share_memory_())
                        else:
                            buffer_tensor = (torch.empty(**spec).to(
                                torch.device('cpu')).share_memory_())

                        position_buffers[key].append(buffer_tensor)

                buffers[device][position] = position_buffers

        return buffers

    def get_action(
        self,
        worker_id: int,
        free_queue: Dict[str, mp.Queue],
        full_queue: Dict[str, mp.Queue],
        model: DouDiZhuModel,
        buffers: Dict[str, Dict[str, List[torch.Tensor]]],
        device: Union[str, int],
    ) -> None:
        """Actor process that interacts with the environment and fills buffers
        with data.

        Args:
            worker_id: Process index for the actor.
            free_queue: Queue for getting free buffer indices.
            full_queue: Queue for passing filled buffer indices to the main process.
            device: Device name ('cpu' or 'cuda:x') where this actor will run.
        """

        try:
            logger.info('Device %s Actor %i started.', str(device), worker_id)

            env = DouDiZhuEnv(objective=self.args.objective)
            env: EnvWrapper = EnvWrapper(env, device)

            done_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            episode_return_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            target_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            obs_x_no_action_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            obs_action_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            obs_z_buf = {
                p: deque(maxlen=self.args.rollout_length)
                for p in self.positions
            }
            size = {p: 0 for p in self.positions}

            position, obs, env_output = env.initial()

            while True:
                while True:
                    obs_x_no_action_buf[position].append(
                        env_output['obs_x_no_action'])
                    obs_z_buf[position].append(env_output['obs_z'])

                    with torch.no_grad():
                        agent_output = model.forward(
                            position,
                            obs['z_batch'],
                            obs['x_batch'],
                            training=False,
                            exp_epsilon=self.args.epsilon_greedy,
                        )

                    action_idx = int(
                        agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][action_idx]
                    obs_action_buf[position].append(cards2tensor(action))

                    size[position] += 1

                    position, obs, env_output = env.step(action)

                    if env_output['done']:
                        for p in self.positions:
                            diff = size[p] - len(target_buf[p])
                            if diff > 0:
                                done_buf[p].extend([False] * (diff - 1))
                                done_buf[p].append(True)

                                episode_return = (
                                    env_output['episode_return']
                                    if p == 'landlord' else
                                    -env_output['episode_return'])
                                episode_return_buf[p].extend([0.0] *
                                                             (diff - 1))
                                episode_return_buf[p].append(episode_return)
                                target_buf[p].extend([episode_return] * diff)
                        break

                for p in self.positions:
                    while size[p] >= self.args.rollout_length:
                        index = free_queue[p].get()
                        if index is None:
                            break

                        for t in range(self.args.rollout_length):
                            buffers[p]['done'][index][t, ...] = done_buf[p][t]
                            buffers[p]['episode_return'][index][t, ...] = (
                                episode_return_buf[p][t])
                            buffers[p]['target'][index][t,
                                                        ...] = target_buf[p][t]
                            buffers[p]['obs_x_no_action'][index][t, ...] = (
                                obs_x_no_action_buf[p][t])
                            buffers[p]['obs_action'][index][
                                t, ...] = obs_action_buf[p][t]
                            buffers[p]['obs_z'][index][t,
                                                       ...] = obs_z_buf[p][t]

                        full_queue[p].put(index)

                        done_buf[p].popleft()
                        episode_return_buf[p].popleft()
                        target_buf[p].popleft()
                        obs_x_no_action_buf[p].popleft()
                        obs_action_buf[p].popleft()
                        obs_z_buf[p].popleft()
                        size[p] -= self.args.rollout_length

        except KeyboardInterrupt:
            logger.info('Actor %i stopped manually.', worker_id)
        except Exception as e:
            logger.error(f'Exception in worker process {worker_id}: {str(e)}')
            traceback.print_exc()
            raise e

    def learn(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        position: str,
        batch: Dict[str, torch.Tensor],
        lock: Lock,
    ) -> Dict[str, Union[float, int]]:
        """Perform a single learning (optimization) step for a given position.

        Args:
            model: The learner's model to be optimized.
            optimizer: The optimizer used to update the model.
            position: The position in the game ('landlord', 'landlord_up', or 'landlord_down').
            batch: A batch of experiences from the environment.
            lock: Lock object to synchronize updates across threads.

        Returns:
            Dictionary with statistics about the learning step.
        """
        device = torch.device(f'cuda:{self.args.training_device}'
                              if self.args.training_device != 'cpu' else 'cpu')

        obs_x_no_action = batch['obs_x_no_action'].to(device)
        obs_action = batch['obs_action'].to(device)

        obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
        obs_x = torch.flatten(obs_x, 0, 1)

        obs_z = torch.flatten(batch['obs_z'].to(device), 0, 1).float()
        target = torch.flatten(batch['target'].to(device), 0, 1)

        episode_returns = batch['episode_return'][batch['done']]

        self.mean_episode_return_buf[position].append(
            torch.mean(episode_returns).to(device))

        with lock:
            learner_outputs = model(obs_z, obs_x, return_value=True)
            loss = self.compute_loss(learner_outputs['values'], target)

            stats = {
                f'mean_episode_return_{position}':
                torch.mean(
                    torch.stack([
                        _r for _r in self.mean_episode_return_buf[position]
                    ])).item(),
                f'loss_{position}':
                loss.item(),
            }

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),
                                     self.args.max_grad_norm)
            optimizer.step()

            for actor_model in self.actor_models.values():
                actor_model.get_model(position).load_state_dict(
                    model.state_dict())

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
        return nn.functional.mse_loss(pred_values, target)

    def batch_and_learn(
        self,
        thread_id: int,
        position: str,
        local_lock: threading.Lock,
        position_lock: threading.Lock,
        lock: threading.Lock,
        device: Union[str, int],
    ) -> None:
        while self.frames < self.args.total_frames:
            batch_data = self.get_batch(
                self.free_queue[device][position],
                self.full_queue[device][position],
                self.buffers[device][position],
                local_lock,
            )
            learner_stats = self.learn(
                self.learner_model.get_model(position),
                self.optimizers[position],
                position,
                batch_data,
                position_lock,
            )

            with lock:
                for key in learner_stats:
                    self.stata_info[key] = learner_stats[key]
                self.frames += self.args.rollout_length * self.args.batch_size
                self.position_frames[position] += (self.args.rollout_length *
                                                   self.args.batch_size)

    def train(self) -> None:
        """Main function for training.

        Initializes necessary components such as buffers, models, optimizers,
        and actors, then spawns subprocesses for actors and threads for
        learning. It also handles logging and checkpointing.
        """

        self.frames, self.stata_info, self.position_frames = (
            0,
            {k: 0
             for k in self.stata_info_keys()},
            {
                'landlord': 0,
                'landlord_up': 0,
                'landlord_down': 0
            },
        )
        # Initialize the actor processes
        ctx = mp.get_context('spawn')
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

        # 初始化 free_queue
        for device in self.device_iterator:
            for m in range(self.args.num_buffers):
                for pos in self.positions:
                    self.free_queue[device][pos].put(m)

        # 初始化 threads, locks, position_locks
        threads, locks, position_locks = [], {}
        position_locks = {
            'landlord': threading.Lock(),
            'landlord_up': threading.Lock(),
            'landlord_down': threading.Lock(),
        }

        for device in self.device_iterator:
            locks[device] = {pos: threading.Lock() for pos in self.positions}
            for i in range(self.args.num_threads):
                for position in self.positions:
                    thread = threading.Thread(
                        target=self.batch_and_learn,
                        name=f'batch-and-learn-{i}',
                        args=(
                            i,
                            position,
                            locks[device][position],
                            position_locks[position],
                            device,
                        ),
                    )
                    thread.start()
                    threads.append(thread)

        fps_log = deque(maxlen=24)
        timer = timeit.default_timer
        last_checkpoint_time = timer() - self.args.save_interval * 60

        try:
            while self.frames < self.args.total_frames:
                start_frames, position_start_frames = (
                    self.frames,
                    self.position_frames.copy(),
                )
                start_time = timer()

                time.sleep(5)

                if timer(
                ) - last_checkpoint_time > self.args.save_interval * 60:
                    self.save_checkpoint(self.checkpoint_path)
                    last_checkpoint_time = timer()

                fps = (self.frames - start_frames) / (timer() - start_time)
                fps_log.append(fps)
                fps_avg = np.mean(fps_log)

                position_fps = {
                    k: (self.position_frames[k] - position_start_frames[k]) /
                    (timer() - start_time)
                    for k in self.position_frames
                }
                logger.info(
                    'After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                    self.frames,
                    self.position_frames['landlord'],
                    self.position_frames['landlord_up'],
                    self.position_frames['landlord_down'],
                    fps,
                    fps_avg,
                    position_fps['landlord'],
                    position_fps['landlord_up'],
                    position_fps['landlord_down'],
                    pprint.pformat(self.stata_info),
                )

        except KeyboardInterrupt:
            pass
        finally:
            for thread in threads:
                thread.join()
            logger.info('Training finished after %d frames.', self.frames)
            self.save_checkpoint(self.checkpoint_path)

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoints.

        Args:
            checkpoint_path: Path to save the checkpoint.
        """
        if self.args.disable_checkpoint:
            return
        logger.info('Saving checkpoint to %s', checkpoint_path)
        torch.save(
            {
                'model_state_dict': {
                    k: self.learner_model.get_model(k).state_dict()
                    for k in self.positions
                },
                'optimizer_state_dict':
                {k: self.optimizers[k].state_dict()
                 for k in self.positions},
                'stats': self.stata_info,
                'frames': self.frames,
                'position_frames': self.position_frames,
            },
            checkpoint_path,
        )
        for position in self.positions:
            model_weights_dir = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (
                    self.args.savedir,
                    self.args.project,
                    position + '_weights_' + str(self.frames) + '.ckpt',
                )))
            torch.save(
                self.learner_model.get_model(position).state_dict(),
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
            for key in self.positions:
                self.learner_model.get_model(key).load_state_dict(
                    checkpoint_states['model_state_dict'][key])
                self.optimizers[key].load_state_dict(
                    checkpoint_states['optimizer_state_dict'][key])
                for device in self.device_iterator:
                    self.actor_models[device].get_model(key).load_state_dict(
                        self.learner_model.get_model(key).state_dict())
            self.stata_info = checkpoint_states['stata_info']
            self.frames = checkpoint_states['frames']
            self.position_frames = checkpoint_states['position_frames']
            logger.info(
                f'Resuming preempted job, current stats:\n{self.stata_info}')
