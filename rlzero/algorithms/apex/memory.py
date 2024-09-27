from collections import deque, namedtuple

import numpy as np

Transition = namedtuple('Transition',
                        ('obs', 'reward', 'action', 'next_obs', 'done'))


class PrioritizedReplayBuffer:

    def __init__(self,
                 buffer_size: int,
                 alpha: float = 0.6,
                 beta: float = 0.4):
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=buffer_size)
        self.prior_buffer = deque(maxlen=buffer_size)

    def add(self, obs, action, reward, next_obs, done, prior):
        self.buffer.append(Transition(obs, action, reward, next_obs, done))
        self.prior_buffer.append(prior)

    def sample(self, batch_size):
        prior_prob = np.array(self.prior_buffer)
        prior_prob = prior_prob**self.alpha / np.sum(prior_prob**self.alpha)
        indices = np.random.choice(len(self.buffer), batch_size, p=prior_prob)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * prior_prob[indices])**-self.beta
        weights /= weights.max()

        obs, actions, rewards, next_obs, dones = zip(*samples)
        batch = dict(
            states=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_obs),
            dones=np.array(dones),
        )
        return indices, batch, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.prior_buffer[idx] = priority

    def __len__(self):
        return len(self.buffer)
