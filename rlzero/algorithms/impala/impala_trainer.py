import copy
import logging
import timeit
from typing import Dict, List

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from rlzero.algorithms.rl_args import RLArguments


def make_env(
    env_id: str,
    seed: int = 42,
    capture_video: bool = False,
    save_video_dir: str = 'work_dir',
    save_video_name: str = 'test',
) -> RecordEpisodeStatistics:
    if capture_video:
        env = gym.make(env_id, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env,
                                       f'{save_video_dir}/{save_video_name}')
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)
    return env


class QNet(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super(QNet, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )
        self.target_qnet = copy.deepcopy(self.q_net)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.q_net(obs)

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action from the actor network."""
        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        q_values = self.forward(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a learning step.

        Args:
            batch (Dict[str, torch.Tensor]): Batch of experience.

        Returns:
            float: Loss value.
        """
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward']
        done = batch['done']

        action = action.to(self.device, dtype=torch.long)
        # Compute current Q values
        current_q_values = self.q_net(obs).gather(1, action)
        # Compute target Q values
        with torch.no_grad():
            greedy_action = self.q_net(next_obs).max(dim=1, keepdim=True)[1]
            next_q_values = self.target_qnet(next_obs).gather(1, greedy_action)

        target_q_values = reward + (1 - done) * 0.95 * next_q_values
        loss = F.mse_loss(current_q_values, target_q_values, reduction='mean')

        return loss


class ImpalaTrainer:

    def __init__(self, args: RLArguments) -> None:
        self.args: RLArguments = args
        self.setup_device()
        self.env: gym.Env = make_env(self.args.env_id)
        self.actor_model = self.setup_model()
        self.learner_model = self.setup_model()
        self.optimizer = self.setup_optimizer()
        self.buffers = self.create_buffers()
        self.global_step = 0

    def setup_device(self):
        if self.args.use_cuda and torch.cuda.is_available():
            logging.info('Using CUDA.')
            self.args.device = torch.device('cuda')
        else:
            logging.info('Not using CUDA.')
            self.args.device = torch.device('cpu')

    def setup_model(self):
        return QNet(
            self.env.observation_space.shape,
            self.env.action_space.n,
        ).to(device=self.args.device)

    def setup_optimizer(self):
        optimizer = torch.optim.RMSprop(
            self.learner_model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            eps=self.args.epsilon,
            alpha=self.args.alpha,
        )

        return optimizer

    def create_buffers(self) -> dict[str, list]:
        specs = dict(
            obs=dict(
                size=(self.args.rollout_length + 1,
                      *self.env.observation_space.shape),
                dtype=torch.float32,
            ),
            next_obs=dict(
                size=(self.args.rollout_length + 1,
                      *self.env.observation_space.shape),
                dtype=torch.float32,
            ),
            reward=dict(size=(self.args.rollout_length + 1, ),
                        dtype=torch.float32),
            done=dict(size=(self.args.rollout_length + 1, ), dtype=torch.bool),
            action=dict(size=(self.args.rollout_length + 1, ),
                        dtype=torch.int64),
        )
        buffers = {key: [] for key in specs}
        for _ in range(self.args.num_buffers):
            for key, spec in specs.items():
                buffer_tensor = torch.empty(**spec).share_memory_()
                buffers[key].append(buffer_tensor)
        return buffers

    def get_action(
        self,
        args,
        actor_index: int,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        actor_model: torch.nn.Module,
        buffers: Dict[str, List[torch.Tensor]],
    ) -> None:
        try:
            logging.info('Actor %i started.', actor_index)
            gym_env = make_env(self.args.env_id)
            obs, _ = gym_env.reset()
            done = False
            while True:
                index = free_queue.get()
                if index is None:
                    break

                for t in range(args.rollout_length):
                    action = actor_model.get_action(obs)
                    next_obs, reward, terminated, truncated, info = gym_env.step(
                        action)
                    done = terminated or truncated
                    buffers['obs'][index][t, ...] = obs
                    buffers['next_obs'][index][t, ...] = next_obs
                    buffers['reward'][index][t, ...] = reward
                    buffers['done'][index][t, ...] = done
                    buffers['action'][index][t, ...] = action

                full_queue.put(index)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logging.error('Exception in worker process %i', actor_index)
            raise e

    def get_batch(
        self,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        buffers,
    ) -> dict[str, torch.Tensor]:
        indices = [full_queue.get() for _ in range(self.args.batch_size)]
        batch = {
            key: torch.stack([buffers[key][m] for m in indices], dim=1)
            for key in buffers
        }
        batch = {
            k: t.to(device=self.args.device, non_blocking=True)
            for k, t in batch.items()
        }
        for m in indices:
            free_queue.put(m)

        return batch

    def learn(
        self,
        actor_model: nn.Module,
        learn_model: nn.Module,
        batch: Dict[str, torch.Tensor],
    ):
        loss = learn_model.learn(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        actor_model.load_state_dict(self.model.state_dict())
        return loss

    def train(self) -> None:
        actor_processes = []
        ctx = mp.get_context('fork')
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for actor_index in range(self.args.num_actors):
            actor = ctx.Process(
                target=self.get_action,
                args=(
                    self.args,
                    actor_index,
                    free_queue,
                    full_queue,
                    self.model,
                    self.buffers,
                ),
            )
            actor.start()
            actor_processes.append(actor)

        for m in range(self.args.num_buffers):
            free_queue.put(m)
        try:
            while self.global_step < self.args.total_steps:
                start_step = self.global_step
                start_time = timeit.default_timer()
                batch = self.get_batch(free_queue, full_queue)
                loss = self.learn(
                    self.actor_model,
                    self.learner_model,
                    batch,
                )

                self.global_step += self.args.rollout_length * self.args.batch_size

                sps = (self.global_step -
                       start_step) / (timeit.default_timer() - start_time)
                logging.info(
                    f'Steps {self.global_step} @ {sps:.1f} SPS. Loss {loss:.6f}'
                )

        except KeyboardInterrupt:
            return  # Try joining actors then quit.
        finally:
            for _ in range(self.args.num_actors):
                free_queue.put(None)
            for actor in actor_processes:
                actor.join(timeout=1)


if __name__ == '__main__':
    args = RLArguments()  # 假设您有一个参数类
    trainer = ImpalaTrainer(args)
    trainer.train()
