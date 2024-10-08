import os
import time
import timeit
import traceback
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import torch
from torch import multiprocessing as mp
from torch import nn

from rlzero.algorithms.impala.atari_wrapper import wrap_deepmind
from rlzero.algorithms.impala.torch_envwrapper import TorchEnvWrapper
from rlzero.algorithms.impala.utils import (compute_baseline_loss,
                                            compute_entropy_loss,
                                            compute_policy_gradient_loss)
from rlzero.algorithms.impala.vtrace import from_logits
from rlzero.algorithms.rl_args import RLArguments
from rlzero.models.atari_model import AtariNet
from rlzero.utils.logger_utils import get_logger
from rlzero.utils.profile import Timings

logger = get_logger('impala_atari')


def create_env(env_id: str) -> gym.Env:
    """Create and wrap an Atari environment.

    Args:
        env_id (str): The ID of the Atari environment.

    Returns:
        gym.Env: The wrapped Atari environment.
    """
    env = gym.make(env_id)
    env = wrap_deepmind(env, frame_stack=4, clip_rewards=False)
    return env


class ImpalaTrainer:
    """Trainer class for IMPALA (Importance Weighted Actor-Learner
    Architecture) algorithm."""

    def __init__(self, args: RLArguments) -> None:
        """Initialize the IMPALA trainer.

        Args:
            args (RLArguments): Configuration arguments for the trainer.
        """
        self.args: RLArguments = args
        self.setup_device()
        self.env: gym.Env = create_env(args.env_id)
        self.actor_model: AtariNet = AtariNet(
            self.env.observation_space.shape,
            self.env.action_space.n.item(),
            use_lstm=args.use_lstm,
        )
        self.actor_model.share_memory()
        self.learner_model: AtariNet = AtariNet(
            self.env.observation_space.shape,
            self.env.action_space.n.item(),
            use_lstm=args.use_lstm,
        )
        self.learner_model.share_memory()
        self.optimizer = self.setup_optimizer()
        self.buffers = self.create_buffers(
            obs_shape=self.env.observation_space.shape,
            num_actions=self.env.action_space.n.item(),
        )
        self.rnn_state_buffers = self.create_rnn_state_buffers()

        if args.num_buffers is None:  # Set sensible default for num_buffers.
            args.num_buffers = max(2 * args.num_actors, args.batch_size)
        if args.num_actors >= args.num_buffers:
            raise ValueError('num_buffers should be larger than num_actors')
        if args.num_buffers < args.batch_size:
            raise ValueError('num_buffers should be larger than batch_size')
        args.checkpoint_path = os.path.join('./work_dir', args.project,
                                            args.algo_name)
        os.makedirs(args.checkpoint_path, exist_ok=True)
        self.args: RLArguments = args
        self.global_step = 0

    def setup_device(self) -> None:
        """Set up the device (CPU or GPU) based on the arguments."""
        if self.args.use_cuda and torch.cuda.is_available():
            logger.info('Using CUDA.')
            self.args.device = torch.device('cuda')
        else:
            logger.info('Not using CUDA.')
            self.args.device = torch.device('cpu')

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Set up the optimizer for the learner model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.RMSprop(
            self.learner_model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            eps=self.args.epsilon,
            alpha=self.args.alpha,
        )
        return optimizer

    def create_rnn_state_buffers(self) -> List[Tuple[torch.Tensor, ...]]:
        """Create initial RNN state buffers.

        Returns:
            List[Tuple[torch.Tensor, ...]]: List of initial RNN states.
        """
        agent_rnn_state_buffers = []
        for _ in range(self.args.num_buffers):
            state = self.actor_model.initial_hidden_state(batch_size=1)
            for t in state:
                t.share_memory_()
            agent_rnn_state_buffers.append(state)
        return agent_rnn_state_buffers

    def create_buffers(self, obs_shape: Tuple[int, ...],
                       num_actions: int) -> Dict[str, List[torch.Tensor]]:
        """Create buffers for storing rollout data.

        Args:
            obs_shape (Tuple[int, ...]): Shape of the observation space.
            num_actions (int): Number of possible actions.

        Returns:
            Dict[str, List[torch.Tensor]]: Buffers for storing rollout data.
        """
        seq_len = self.args.rollout_length
        specs = dict(
            obs=dict(size=(seq_len + 1, *obs_shape), dtype=torch.uint8),
            reward=dict(size=(seq_len + 1, ), dtype=torch.float32),
            done=dict(size=(seq_len + 1, ), dtype=torch.bool),
            last_action=dict(size=(seq_len + 1, ), dtype=torch.int64),
            action=dict(size=(seq_len + 1, ), dtype=torch.int64),
            episode_return=dict(size=(seq_len + 1, ), dtype=torch.float32),
            episode_step=dict(size=(seq_len + 1, ), dtype=torch.int32),
            policy_logits=dict(size=(seq_len + 1, num_actions),
                               dtype=torch.float32),
            baseline=dict(size=(seq_len + 1, ), dtype=torch.float32),
        )
        buffers = {key: [] for key in specs}
        for _ in range(self.args.num_buffers):
            for key, spec in specs.items():
                buffer_tensor = torch.empty(**spec).share_memory_()
                buffers[key].append(buffer_tensor)
        return buffers

    def get_action(
        self,
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module,
        buffers: Dict[str, List[torch.Tensor]],
        rnn_state_buffers: List[Tuple[torch.Tensor, ...]],
    ) -> None:
        """Actor process that collects rollout data.

        Args:
            actor_index (int): Index of the actor.
            free_queue (mp.SimpleQueue): Queue for free buffer indices.
            full_queue (mp.SimpleQueue): Queue for full buffer indices.
            actor_model (torch.nn.Module): The actor model.
            buffers (Dict[str, List[torch.Tensor]]): Buffers for storing rollout data.
            rnn_state_buffers (List[Tuple[torch.Tensor, ...]]): Initial RNN states.
        """
        try:
            logger.info('Actor %i started.', actor_index)
            timings = Timings()  # Keep track of how fast things are.
            gym_env = create_env(self.args.env_id)
            env: TorchEnvWrapper = TorchEnvWrapper(gym_env)
            env_output = env.reset()
            agent_state = self.actor_model.initial_hidden_state(batch_size=1)
            agent_output, unused_state = self.actor_model(
                env_output, agent_state)
            while True:
                index = free_queue.get()
                if index is None:
                    break

                # Write old rollout end.
                for key in env_output:
                    buffers[key][index][0, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][0, ...] = agent_output[key]
                for i, tensor in enumerate(agent_state):
                    rnn_state_buffers[index][i][...] = tensor

                for t in range(self.args.rollout_length):
                    timings.reset()
                    with torch.no_grad():
                        agent_output, agent_state = actor_model(
                            env_output, agent_state)
                    timings.time('model')
                    env_output = env.step(agent_output['action'])
                    timings.time('step')
                    for key in env_output:
                        buffers[key][index][t + 1, ...] = env_output[key]
                    for key in agent_output:
                        buffers[key][index][t + 1, ...] = agent_output[key]

                    timings.time('write')

                full_queue.put(index)

            if actor_index == 0:
                logger.info('Actor %i: %s', actor_index, timings.summary())

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logger.error('Exception in worker process %i: %s', actor_index,
                         str(e))
            traceback.print_exc()
            raise e

    def get_batch(
        self,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers: Dict[str, List[torch.Tensor]],
        rnn_state_buffers: List[Tuple[torch.Tensor, ...]],
        timings: Timings,
        lock: mp.Lock = mp.Lock(),
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Get a batch of rollout data for training.

        Args:
            free_queue (mp.SimpleQueue): Queue for free buffer indices.
            full_queue (mp.SimpleQueue): Queue for full buffer indices.
            buffers (Dict[str, List[torch.Tensor]]): Buffers for storing rollout data.
            rnn_state_buffers (List[Tuple[torch.Tensor, ...]]): Initial RNN states.
            timings (Timings): Timings object to track performance.
            lock (mp.Lock): Lock for thread safety.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]: Batch of rollout data and initial RNN states.
        """
        with lock:
            timings.time('lock')
            indices = [full_queue.get() for _ in range(self.args.batch_size)]
            timings.time('dequeue')
        batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1)
            for key in buffers
        }
        initial_rnn_state = (torch.cat(
            ts, dim=1) for ts in zip(*[rnn_state_buffers[m] for m in indices]))
        timings.time('batch')
        for m in indices:
            free_queue.put(m)
        timings.time('enqueue')

        batch = {
            k: t.to(device=self.args.device, non_blocking=True)
            for k, t in batch.items()
        }
        initial_rnn_state = tuple(
            t.to(device=self.args.device, non_blocking=True)
            for t in initial_rnn_state)
        timings.time('device')

        return batch, initial_rnn_state

    def learn(
            self,
            actor_model: nn.Module,
            learner_model: nn.Module,
            batch: Dict[str, torch.Tensor],
            initial_rnn_state: Tuple[torch.Tensor, ...],
            lock: mp.Lock = mp.Lock(),
    ) -> Dict[str, Any]:
        """Perform a learning step using the batch of rollout data.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of rollout data.
            initial_rnn_state (Tuple[torch.Tensor, ...]): Initial RNN states.
            lock (mp.Lock): Lock for thread safety.

        Returns:
            Dict[str, Any]: Statistics from the learning step.
        """
        with lock:
            learner_outputs, unused_state = learner_model(
                batch, initial_rnn_state)

            # Take final value function slice for bootstrapping.
            bootstrap_value = learner_outputs['baseline'][-1]

            # Move from obs[t] -> action[t] to action[t] -> obs[t].
            batch = {key: tensor[1:] for key, tensor in batch.items()}
            learner_outputs = {
                key: tensor[:-1]
                for key, tensor in learner_outputs.items()
            }

            rewards = batch['reward']
            if self.args.reward_clipping == 'abs_one':
                clipped_rewards = torch.clamp(rewards, -1, 1)
            elif self.args.reward_clipping == 'none':
                clipped_rewards = rewards

            discounts = (~batch['done']).float() * self.args.discounting

            vtrace_returns = from_logits(
                behavior_policy_logits=batch['policy_logits'],
                target_policy_logits=learner_outputs['policy_logits'],
                actions=batch['action'],
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs['baseline'],
                bootstrap_value=bootstrap_value,
            )

            pg_loss = compute_policy_gradient_loss(
                learner_outputs['policy_logits'],
                batch['action'],
                vtrace_returns.pg_advantages,
            )
            baseline_loss = self.args.baseline_cost * compute_baseline_loss(
                vtrace_returns.vs - learner_outputs['baseline'])
            entropy_loss = self.args.entropy_cost * compute_entropy_loss(
                learner_outputs['policy_logits'])

            total_loss = pg_loss + baseline_loss + entropy_loss

            episode_returns = batch['episode_return'][batch['done']]
            stats = {
                'episode_returns': tuple(episode_returns.cpu().numpy()),
                'mean_episode_return': torch.mean(episode_returns).item(),
                'total_loss': total_loss.item(),
                'pg_loss': pg_loss.item(),
                'baseline_loss': baseline_loss.item(),
                'entropy_loss': entropy_loss.item(),
            }

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(learner_model.parameters(),
                                     self.args.max_grad_norm)
            self.optimizer.step()

            actor_model.load_state_dict(learner_model.state_dict())
            return stats

    def learn_process(
            self,
            threading_id: int,
            actor_model: nn.Module,
            learner_model: nn.Module,
            free_queue: mp.SimpleQueue,
            full_queue: mp.SimpleQueue,
            buffers: Dict[str, List[torch.Tensor]],
            rnn_state_buffers: List[Tuple[torch.Tensor, ...]],
            lock: mp.Lock = mp.Lock(),
    ) -> None:
        """Thread target for the learning process.

        Args:
            threading_id (int): ID of the thread.
            free_queue (mp.SimpleQueue): Queue for free buffer indices.
            full_queue (mp.SimpleQueue): Queue for full buffer indices.
            buffers (Dict[str, List[torch.Tensor]]): Buffers for storing rollout data.
            rnn_state_buffers (List[Tuple[torch.Tensor, ...]]): Initial RNN states.
            lock (mp.Lock): Lock for thread safety.
        """

        try:
            timings = Timings()
            while self.global_step < self.args.total_steps:
                timings.reset()
                batch, agent_state = self.get_batch(
                    free_queue,
                    full_queue,
                    buffers,
                    rnn_state_buffers,
                    timings,
                )
                stats = self.learn(actor_model, learner_model, batch,
                                   agent_state, lock)
                timings.time('learn')
                with lock:
                    to_log = dict(step=self.global_step)
                    to_log.update({k: stats[k] for k in self.stat_keys})
                    logger.info(to_log)
                    self.global_step += self.args.rollout_length * self.args.batch_size

            if threading_id == 0:
                logger.info('Batch and learn: %s', timings.summary())
        except KeyboardInterrupt:
            return
        except Exception as e:
            logger.error('Exception in learn process %i: %s', threading_id,
                         str(e))
            traceback.print_exc()
            raise e

    def train(self) -> None:
        """Main training loop for the IMPALA algorithm."""
        self.stat_keys = [
            'total_loss',
            'mean_episode_return',
            'pg_loss',
            'baseline_loss',
            'entropy_loss',
        ]
        checkpoint_path = os.path.join(self.args.output_dir, self.args.project,
                                       'model.tar')

        actor_processes = []
        ctx = mp.get_context('fork')
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for actor_index in range(self.args.num_actors):
            actor = mp.Process(
                target=self.get_action,
                name=f'actor-rpocess-%={actor_index}',
                args=(
                    actor_index,
                    free_queue,
                    full_queue,
                    self.actor_model,
                    self.buffers,
                    self.rnn_state_buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

        for m in range(self.args.num_buffers):
            free_queue.put(m)

        learner_process = []
        for learner_id in range(self.args.num_learners):
            learner = mp.Process(
                target=self.learn_process,
                name=f'learn-process-{learner_id}',
                args=(
                    learner_id,
                    self.actor_model,
                    self.learner_model,
                    free_queue,
                    full_queue,
                    self.buffers,
                    self.rnn_state_buffers,
                    mp.Lock(),
                ),
            )
            learner.start()
            learner_process.append(learner)

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer()
            while self.global_step < self.args.total_steps:
                start_step = self.global_step
                start_time = timer()
                time.sleep(5)

                if timer(
                ) - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                    self.save_checkpoint(checkpoint_path)
                    last_checkpoint_time = timer()
                sps = (self.global_step - start_step) / (timer() - start_time)
                logger.info('Steps %i @ %.1f SPS.', self.global_step, sps)

        except KeyboardInterrupt:
            return  # Try joining actors then quit.
        else:
            for learner in learner_process:
                learner.join(timeout=1)
            logger.info('Learning finished after %d steps.', self.global_step)
        finally:
            for _ in range(self.args.num_actors):
                free_queue.put(None)
            for actor in actor_processes:
                actor.join(timeout=1)
            if actor.is_active():
                actor.terminate()

            for learner in learner_process:
                learner.join(timeout=1)
                if learner.is_alive():
                    logger.warning(
                        'Learner process did not terminate, force terminating...'
                    )
                    learner.terminate()
                logger.info('All processes have been stopped.')

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save the current state of the model and optimizer.

        Args:
            checkpoint_path (str): Path to save the checkpoint.
        """
        if self.args.disable_checkpoint:
            return
        logger.info('Saving checkpoint to %s', checkpoint_path)
        try:
            torch.save(
                {
                    'model_state_dict': self.actor_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'hparam': vars(self.args),
                },
                checkpoint_path,
            )
        except Exception as e:
            logger.error('Failed to save checkpoint: %s', str(e))  # 记录保存失败的错误


if __name__ == '__main__':
    args = RLArguments()  # 假设您有一个参数类
    trainer = ImpalaTrainer(args)
    trainer.train()
