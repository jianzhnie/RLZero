import multiprocessing as mp
import random
import traceback
from collections import deque
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import RecordEpisodeStatistics

from rlzero.utils.logger_utils import get_logger

logger = get_logger(__name__)


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


class QNetwork(nn.Module):
    """A simple feedforward neural network for Q-learning.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor representing the Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """A replay buffer to store and sample experiences.

    Args:
        buffer_size (int): Maximum size of the buffer.
    """

    def __init__(self, buffer_size: int):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        """Add an experience to the buffer.

        Args:
            experience (Tuple[np.ndarray, int, float, np.ndarray, bool]): A tuple containing (state, action, reward, next_state, done).
        """
        self.memory.append(experience)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing states, actions, rewards, next_states, and dones.
        """
        experiences = random.sample(self.memory, k=batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        batch = dict(states=np.array(states),
                     actions=np.array(actions),
                     rewards=np.array(rewards),
                     next_states=np.array(next_states),
                     dones=np.array(dones))
        return batch

    def __len__(self) -> int:
        """Get the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self.memory)


class ImpalaDQN:
    """Implements the IMPALA (Importance Weighted Actor-Learner Architecture)
    with DQN.

    Args:
        state_dim (int): Dimension of the state space.
        action_dim (int): Dimension of the action space.
        num_actors (int): Number of actor processes.
        buffer_size (int): Maximum size of the replay buffer.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Number of experiences to sample for training.
        lr (float): Learning rate for the optimizer.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_actors: int = 4,
                 max_timesteps: int = 10000,
                 buffer_size: int = 10000,
                 eps_greedy: float = 0.1,
                 target_update_frequency: int = 1000,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 lr: float = 0.001,
                 device: str = 'cpu') -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actors = num_actors
        self.max_timesteps = max_timesteps
        self.buffer_size = buffer_size
        self.eps_greedy = eps_greedy
        self.target_update_frequency = target_update_frequency
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.data_queue = mp.Queue(maxsize=100)
        self.param_queue = mp.Queue()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the actor network.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            np.ndarray: Selected action.
        """
        # epsilon greedy policy
        if np.random.rand() <= self.eps_greedy:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = self.predict(obs)

        return action

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs (np.ndarray): Current observation.

        Returns:
            int: Selected action.
        """
        if obs.ndim == 1:
            # Expand to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)

        obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        with torch.no_grad():
            q_values = self.q_network(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action

    def actor_process(self, actor_id: int, env: gym.Env, data_queue: mp.Queue,
                      stop_event: mp.Event):
        """Actor process that interacts with the environment and collects
        experiences.

        Args:
            actor_id (int): ID of the actor.
            env (gym.Env): Environment to interact with.
            data_queue (mp.Queue): Queue to send collected experiences to the learner.
            stop_event (mp.Event): Event to signal the actor to stop.
        """
        logger.info(f'Actor {actor_id} started')
        try:
            while not stop_event.is_set():
                state, _ = env.reset()
                buffer: List[Tuple[np.ndarray, int, float, np.ndarray,
                                   bool]] = []
                done = False
                while not done:
                    action = self.get_action(state)
                    next_state, reward, terminal, truncated, info = env.step(
                        action)
                    done = terminal or truncated
                    if info and 'episode' in info:
                        info_item = {
                            k: v.item()
                            for k, v in info['episode'].items()
                        }
                        episode_reward = info_item['r']
                        episode_step = info_item['l']

                    buffer.append((state, action, reward, next_state, done))
                    state = next_state
                if buffer:
                    data_queue.put(buffer)

                if actor_id == 0:
                    logger.info(
                        'Actor {}: , episode step: {}, episode reward: {}'.
                        format(actor_id, episode_step, episode_reward), )

        except Exception as e:
            logger.error(f'Exception in actor process {actor_id}: {e}')
            traceback.print_exc()

    def learner_process(self, data_queue: mp.Queue, stop_event: mp.Event):
        """Learner process that trains the Q-network using experiences from the
        actors.

        Args:
            data_queue (mp.Queue): Queue to receive experiences from actors.
            stop_event (mp.Event): Event to signal the learner to stop.
        """

        try:
            while self.global_step < self.max_timesteps and not stop_event.is_set(
            ):
                try:
                    # Non-blocking with timeout
                    data = data_queue.get(timeout=0.1)
                except data_queue.Empty:
                    continue  # 如果队列为空，继续循环

                actor_step = len(data)
                self.global_step += actor_step
                for experience in data:
                    self.replay_buffer.add(experience)

                if len(self.replay_buffer) >= self.batch_size:
                    batch = self.replay_buffer.sample(self.batch_size)

                    states = torch.tensor(batch['states'], dtype=torch.float32)
                    actions = torch.tensor(batch['actions'],
                                           dtype=torch.long).unsqueeze(1)
                    rewards = torch.tensor(batch['rewards'],
                                           dtype=torch.float32).unsqueeze(1)
                    next_states = torch.tensor(batch['next_states'],
                                               dtype=torch.float32)
                    dones = torch.tensor(batch['dones'],
                                         dtype=torch.float32).unsqueeze(1)

                    with torch.no_grad():
                        next_q_values = self.target_network(next_states).max(
                            1, keepdim=True)[0]
                        expected_q_values = rewards + self.gamma * next_q_values * (
                            1 - dones)

                    q_values = self.q_network(states).gather(1, actions)
                    loss = (q_values - expected_q_values).pow(2).mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    logger.info(
                        f'global_step: {self.global_step}, loss: {loss.item()}'
                    )

                if self.global_step % self.target_update_frequency == 0:
                    self.target_network.load_state_dict(
                        self.q_network.state_dict())

        except Exception as e:
            logger.error(f'Exception in learner process: {e}')
        finally:
            logger.info('Learner process is shutting down')

    def evaluate(self, n_eval_episodes: int = 5) -> dict[str, float]:
        eval_rewards = []
        eval_steps = []
        for _ in range(n_eval_episodes):
            obs, info = self.test_env.reset()
            done = False
            episode_reward = 0.0
            episode_step = 0
            while not done:
                action = self.predict(obs)
                next_obs, reward, terminated, truncated, info = self.test_env.step(
                    action)
                obs = next_obs
                done = terminated or truncated
                if info and 'episode' in info:
                    info_item = {
                        k: v.item()
                        for k, v in info['episode'].items()
                    }
                    episode_reward = info_item['r']
                    episode_step = info_item['l']
                if done:
                    self.test_env.reset()
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_step)

        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_steps),
            'length_std': np.std(eval_steps),
        }

    def run(self) -> None:
        """Run the IMPALA DQN algorithm."""
        stop_event = mp.Event()
        actor_processes = []
        for i in range(self.num_actors):
            train_env = make_env(env_id='CartPole-v1')
            actor = mp.Process(target=self.actor_process,
                               args=(i, train_env, self.data_queue,
                                     stop_event))
            actor.start()
            actor_processes.append(actor)

        learner = mp.Process(target=self.learner_process,
                             args=(self.data_queue, stop_event))
        learner.start()

        try:
            learner.join()
        except KeyboardInterrupt:
            logger.info(
                'Keyboard interrupt received, stopping all processes...')
        finally:
            stop_event.set()
            for actor in actor_processes:
                actor.join(timeout=1)  # 给予一定的时间让进程正常结束
                if actor.is_alive():
                    logger.warning(
                        f'Actor process {actor.pid} did not terminate, force terminating...'
                    )
                    actor.terminate()
            learner.join(timeout=1)
            if learner.is_alive():
                logger.warning(
                    'Learner process did not terminate, force terminating...')
                learner.terminate()
            logger.info('All processes have been stopped.')


if __name__ == '__main__':
    impala_dqn = ImpalaDQN(state_dim=4, action_dim=2, num_actors=1)
    impala_dqn.run()
