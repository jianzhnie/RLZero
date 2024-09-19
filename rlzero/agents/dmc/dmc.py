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
from typing import Any, Dict, Iterator, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from rlzero.agents.dmc.env_utils import EnvWrapper
from rlzero.agents.dmc.utils import cards2tensor
from rlzero.agents.rl_args import RLArguments
from rlzero.envs.doudizhu.env import DouDizhuEnv
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

    def __init__(
        self,
        positions: List[str] = ['landlord', 'landlord_up', 'landlord_down'],
        args: RLArguments = RLArguments,
    ) -> None:
        """Initialize the DMCAgent.

        Args:
            actor_models: Dictionary of actor models for each position.
            optimizers: Dictionary of optimizers for each position.
            positions: List of positions in the game.
            args: Command-line arguments.
        """
        self.mean_episode_return_buf = {
            p: deque(maxlen=100)
            for p in positions
        }
        self.positions = positions
        self.doudizhu_model = DouDiZhuModel(device=args.training_device)
        self.learner_model = DouDiZhuModel(device=args.training_device)
        self.optimizers = self.create_optimizers(args.learning_rate,
                                                 args.momentum, args.epsilon,
                                                 args.alpha)
        self.args: RLArguments = args

    def create_optimizers(
        self,
        learning_rate: float,
        momentum: float,
        epsilon: float,
        alpha: float,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Create optimizers for each position.

        Args:
            learning_rate: Learning rate for the optimizer.
            momentum: Momentum for the optimizer.
            epsilon: Epsilon for the optimizer.
            alpha: Alpha for the optimizer.

        Returns:
            Dictionary of optimizers for each position.
        """
        optimizers = {}

        for position in self.positions:
            position_parameters = getattr(self.learner_model,
                                          position).parameters()
            optimizer = torch.optim.RMSprop(
                position_parameters,
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
        rollout_length: int,
        num_buffers: int,
        device_iterator: Iterator[int],
    ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
        """Create buffers for each position and device.

        Args:
            rollout_length: Length of the rollout.
            num_buffers: Number of buffers to create.
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

    def act(
        self,
        worker_id: int,
        objective: str,
        rollout_length: int,
        exp_epsilon: float,
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
            objective: Objective of the actor (e.g., wp/adp/logadp).
            rollout_length: Length of the rollout.
            exp_epsilon: Exploration epsilon for the actor.
            free_queue: Queue for getting free buffer indices.
            full_queue: Queue for passing filled buffer indices to the main process.
            model: Model used for decision-making in the game.
            buffers: Shared memory buffers for storing game experiences.
            device: Device name ('cpu' or 'cuda:x') where this actor will run.
        """

        try:
            logger.info('Device %s Actor %i started.', str(device), worker_id)

            env = DouDizhuEnv(objective=objective)
            env: EnvWrapper = EnvWrapper(env, device)

            done_buf = {
                p: deque(maxlen=rollout_length)
                for p in self.positions
            }
            episode_return_buf = {
                p: deque(maxlen=rollout_length)
                for p in self.positions
            }
            target_buf = {
                p: deque(maxlen=rollout_length)
                for p in self.positions
            }
            obs_x_no_action_buf = {
                p: deque(maxlen=rollout_length)
                for p in self.positions
            }
            obs_action_buf = {
                p: deque(maxlen=rollout_length)
                for p in self.positions
            }
            obs_z_buf = {
                p: deque(maxlen=rollout_length)
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
                            exp_epsilon=exp_epsilon,
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
                    while size[p] >= rollout_length:
                        index = free_queue[p].get()
                        if index is None:
                            break

                        for t in range(rollout_length):
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
                        size[p] -= rollout_length

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

        if position not in self.mean_episode_return_buf:
            self.mean_episode_return_buf[position] = deque(
                maxlen=self.args.buffer_size)

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

    def train(
        self,
        rollout_length: int,
        batch_size: int,
    ) -> None:
        """Main function for training.

        Initializes necessary components such as buffers, models, optimizers,
        and actors, then spawns subprocesses for actors and threads for
        learning. It also handles logging and checkpointing.

        Args:
            rollout_length: Length of the rollout.
            batch_size: Size of the batch for learning.
        """
        if not self.args.actor_device_cpu or self.args.training_device != 'cpu':
            if not torch.cuda.is_available():
                raise AssertionError(
                    'CUDA not available. If you have GPUs, specify their IDs with --gpu_devices. '
                    'Otherwise, train with CPU using: python3 train.py --actor_device_cpu --training_device cpu'
                )

        checkpoint_path = os.path.expandvars(
            os.path.expanduser(
                f'{self.args.savedir}/{self.args.xpid}/model.tar'))

        if self.args.actor_device_cpu:
            device_iterator = ['cpu']
        else:
            device_iterator = range(self.args.num_actor_devices)
            assert self.args.num_actor_devices <= len(
                self.args.gpu_devices.split(',')
            ), 'Number of actor devices cannot exceed the available GPU devices'

        models = {
            device: DouDiZhuModel(device=device).share_memory().eval()
            for device in device_iterator
        }
        buffers = self.create_buffers(self.args, device_iterator)

        ctx = mp.get_context('spawn')
        free_queue, full_queue = {}, {}

        for device in device_iterator:
            free_queue[device] = {
                'landlord': ctx.SimpleQueue(),
                'landlord_up': ctx.SimpleQueue(),
                'landlord_down': ctx.SimpleQueue(),
            }
            full_queue[device] = {
                'landlord': ctx.SimpleQueue(),
                'landlord_up': ctx.SimpleQueue(),
                'landlord_down': ctx.SimpleQueue(),
            }

        learner_model = DouDiZhuModel(device=self.args.training_device)
        optimizers = self.create_optimizers(self.args, learner_model)

        # Stat Keys
        stat_keys = [
            'mean_episode_return_landlord',
            'loss_landlord',
            'mean_episode_return_landlord_up',
            'loss_landlord_up',
            'mean_episode_return_landlord_down',
            'loss_landlord_down',
        ]

        frames, stats, position_frames = (
            0,
            {k: 0
             for k in stat_keys()},
            {
                'landlord': 0,
                'landlord_up': 0,
                'landlord_down': 0
            },
        )

        if self.args.load_model and os.path.exists(checkpoint_path):
            checkpoint_states = torch.load(
                checkpoint_path,
                map_location=torch.device(
                    'cuda' if self.args.training_device != 'cpu' else 'cpu'),
            )
            stats = checkpoint_states['stats']
            frames = checkpoint_states['frames']
            position_frames = checkpoint_states['position_frames']
            logger.info(f'Resuming from checkpoint, stats:\n{stats}')

        actor_processes = []
        for device in device_iterator:
            for i in range(self.args.num_actors):
                actor_process = ctx.Process(
                    target=self.act,
                    args=(
                        i,
                        device,
                        free_queue[device],
                        full_queue[device],
                        models[device],
                        buffers[device],
                        self.args,
                    ),
                )
                actor_process.start()
                actor_processes.append(actor_process)

        def batch_and_learn(
                i: int,
                device: Any,
                position: str,
                local_lock: threading.Lock,
                position_lock: threading.Lock,
                lock=threading.Lock(),
        ) -> None:
            nonlocal frames, position_frames, stats
            while frames < self.args.total_frames:
                batch = self.get_batch(
                    free_queue[device][position],
                    full_queue[device][position],
                    buffers[device][position],
                    self.args,
                    local_lock,
                )
                _stats = self.learn(
                    position,
                    models,
                    learner_model.get_model(position),
                    batch,
                    optimizers[position],
                    self.args,
                    position_lock,
                )

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]
                    frames += rollout_length * batch_size
                    position_frames[position] += rollout_length * batch_size

        for device in device_iterator:
            for m in range(self.args.num_buffers):
                for pos in ['landlord', 'landlord_up', 'landlord_down']:
                    free_queue[device][pos].put(m)

        threads, locks, position_locks = (
            [],
            {},
            {
                'landlord': threading.Lock(),
                'landlord_up': threading.Lock(),
                'landlord_down': threading.Lock(),
            },
        )

        for device in device_iterator:
            locks[device] = {
                pos: threading.Lock()
                for pos in ['landlord', 'landlord_up', 'landlord_down']
            }
            for i in range(self.args.num_threads):
                for position in ['landlord', 'landlord_up', 'landlord_down']:
                    thread = threading.Thread(
                        target=batch_and_learn,
                        name=f'batch-and-learn-{i}',
                        args=(
                            i,
                            device,
                            position,
                            locks[device][position],
                            position_locks[position],
                        ),
                    )
                    thread.start()
                    threads.append(thread)

        fps_log = deque(maxlen=24)
        timer = timeit.default_timer
        last_checkpoint_time = timer() - self.args.save_interval * 60

        try:
            while frames < self.args.total_frames:
                start_frames, position_start_frames = frames, position_frames.copy(
                )
                start_time = timer()

                time.sleep(5)

                if timer(
                ) - last_checkpoint_time > self.args.save_interval * 60:
                    self.checkpoint(frames)
                    last_checkpoint_time = timer()

                fps = (frames - start_frames) / (timer() - start_time)
                fps_log.append(fps)
                fps_avg = np.mean(fps_log)

                position_fps = {
                    k: (position_frames[k] - position_start_frames[k]) /
                    (timer() - start_time)
                    for k in position_frames
                }
                logger.info(
                    'After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                    frames,
                    position_frames['landlord'],
                    position_frames['landlord_up'],
                    position_frames['landlord_down'],
                    fps,
                    fps_avg,
                    position_fps['landlord'],
                    position_fps['landlord_up'],
                    position_fps['landlord_down'],
                    pprint.pformat(stats),
                )

        except KeyboardInterrupt:
            pass
        finally:
            for thread in threads:
                thread.join()
            logger.info('Training finished after %d frames.', frames)
            self.checkpoint(frames)

    def checkpoint(self, frames: int, checkpoint_path: str) -> None:
        """Save model checkpoints.

        Args:
            frames: Number of frames processed.
            checkpoint_path: Path to save the checkpoint.
        """
        if self.args.disable_checkpoint:
            return
        logger.info('Saving checkpoint to %s', checkpoint_path)
        torch.save(
            {
                'model_state_dict': {
                    k: self.learner_model.get_model(k).state_dict()
                    for k in ['landlord', 'landlord_up', 'landlord_down']
                },
                'optimizer_state_dict':
                {k: self.optimizers[k].state_dict()
                 for k in self.optimizers},
                'stats': stats,
                'args': vars(self.args),
                'frames': frames,
                'position_frames': position_frames,
            },
            checkpoint_path,
        )
