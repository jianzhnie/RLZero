import multiprocessing as mp
import random
from collections import deque
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
                 max_time_steps: int = 10000,
                 buffer_size: int = 10000,
                 gamma: float = 0.99,
                 batch_size: int = 32,
                 lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actors = num_actors
        self.max_time_steps = max_time_steps
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.data_queue = mp.Queue()
        self.param_queue = mp.Queue()
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.global_step = 0

    def get_action(self, state: np.ndarray) -> int:
        """Select an action based on the current state.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        with torch.no_grad():
            if state.ndim == 1:
                state = np.expand_dims(state, axis=0)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()
        return action

    def actor_process(self, actor_id: int, env: gym.Env, data_queue: mp.Queue,
                      stop_event: mp.Event):
        """Actor process that interacts with the environment and collects
        experiences.

        Args:
            actor_id (int): ID of the actor.
            env (gym.Env): Environment to interact with.
            q_network (QNetwork): Q-network for action selection.
            data_queue (mp.Queue): Queue to send collected experiences to the learner.
            stop_event (mp.Event): Event to signal the actor to stop.
        """
        state, _ = env.reset()
        buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        done = False
        episode_step = 0
        try:
            while not done:
                action = self.get_action(state)
                next_state, reward, terminal, truncated, _ = env.step(action)
                episode_step += 1
                done = terminal or truncated
                buffer.append((state, action, reward, next_state, done))
                state = next_state
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            data_queue.put(buffer)

    def learner_process(self, data_queue: mp.Queue, param_queue: mp.Queue,
                        q_network: QNetwork, target_network: QNetwork,
                        optimizer: optim.Optimizer,
                        replay_buffer: ReplayBuffer, stop_event: mp.Event):
        """Learner process that trains the Q-network using experiences from the
        actors.

        Args:
            data_queue (mp.Queue): Queue to receive experiences from actors.
            param_queue (mp.Queue): Queue to send updated parameters to actors.
            q_network (QNetwork): Q-network to train.
            target_network (QNetwork): Target network for stable training.
            optimizer (optim.Optimizer): Optimizer for training the Q-network.
            replay_buffer (ReplayBuffer): Replay buffer to store experiences.
            stop_event (mp.Event): Event to signal the learner to stop.
        """

        while self.global_step < self.max_time_steps and not stop_event.is_set(
        ):
            if not data_queue.empty():
                data = data_queue.get()
                actor_step = len(data)
                self.global_step += actor_step
                for experience in data:
                    replay_buffer.add(experience)

            if len(replay_buffer) >= self.batch_size:
                batch = replay_buffer.sample(self.batch_size)

                states = torch.tensor(batch['states'], dtype=torch.float32)
                actions = torch.tensor(batch['actions'],
                                       dtype=torch.long).view(-1, 1)
                rewards = torch.tensor(batch['rewards'],
                                       dtype=torch.float32).view(-1, 1)
                next_states = torch.tensor(batch['next_states'],
                                           dtype=torch.float32)
                dones = torch.tensor(batch['dones'],
                                     dtype=torch.float32).view(-1, 1)

                with torch.no_grad():
                    next_q_values = target_network(next_states).max(
                        1, keepdim=True)[0]
                    expected_q_values = rewards + self.gamma * next_q_values * (
                        1 - dones)

                q_values = q_network(states).gather(1, actions)
                loss = (q_values - expected_q_values).pow(2).mean()

                print(f'global_step: {self.global_step}, loss: {loss.item()}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if np.random.rand() < 0.01:
                    target_network.load_state_dict(q_network.state_dict())

                param_queue.put(q_network.state_dict())

    def run(self) -> None:
        """Run the IMPALA DQN algorithm."""
        stop_event = mp.Event()
        actors = []
        for i in range(self.num_actors):
            env = gym.make('CartPole-v1')
            actor = mp.Process(target=self.actor_process,
                               args=(i, env, self.data_queue, stop_event))
            actor.daemon = True
            actors.append(actor)
            actor.start()

        learner = mp.Process(target=self.learner_process,
                             args=(self.data_queue, self.param_queue,
                                   self.q_network, self.target_network,
                                   self.optimizer, self.replay_buffer,
                                   stop_event))
        learner.start()

        try:
            for actor in actors:
                actor.join()
            learner.join()
        except KeyboardInterrupt:
            stop_event.set()
        except Exception as e:
            print(f'An error occurred: {e}')


if __name__ == '__main__':
    impala_dqn = ImpalaDQN(state_dim=4, action_dim=2)
    impala_dqn.run()