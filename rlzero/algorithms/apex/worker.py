import random

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim

from rlzero.algorithms.apex.network import QNet


class Actor(mp.Process):

    def __init__(self, actor_id, env_name, replay_buffer, eps=0.4, gamma=0.99):
        self.actor_id = actor_id
        self.env = gym.make(env_name)
        self.action_dim = self.env.action_space.n
        self.obs_dim = self.env.action_space.shape[0]
        self.q_net = QNet(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=128,
        )
        self.eps_greedy = 0.1
        self.replay_buffer = replay_buffer
        self.eps = eps
        self.gamma = gamma

    def run(self) -> None:
        state = self.env.reset()
        done = False
        while not done:
            if np.random.rand() < self.eps:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add((state, action, reward, next_state, done))
            state = next_state

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
            q_values = self.q_net(obs)
        action = torch.argmax(q_values, dim=1).item()
        return action


class Learner:

    def __init__(self,
                 model,
                 target_model,
                 replay_buffer,
                 batch_size=32,
                 gamma=0.99,
                 lr=1e-3):
        self.model = model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        indices, samples, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        td_error = torch.abs(current_q_values -
                             target_q_values).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_error)

        loss = (weights *
                (current_q_values - target_q_values.detach())**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
