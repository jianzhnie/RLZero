"""本项目是一个极简的MuZero的实现，没有使用MCTS方法，模型由Representation_model、Dynamics_Model、Predict
ion_Model构成：

Representation_model将一组观察值映射到神经网络的隐藏状态s； 动态Dynamics_Model根据动作a_(t + 1)将状态s_t映射到下一个状态s_(t +
1)，同时估算在此过程的回报r_t，这样模型就能够不断向前扩展； 预测Prediction_Model 根据状态s_t对策略p_t和值v_t进行估计；
"""

## baidu muzero notebook
import copy
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from tqdm import tqdm


class Dynamics_Model(nn.Module):
    # action encoding - one hot
    def __init__(self, hidden_dim, action_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        network = [
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, hidden_dim)
        ]
        self.network = nn.Sequential(*network)
        self.hs = nn.Linear(hidden_dim, hidden_dim)
        self.r = nn.Linear(hidden_dim, 1)

    def forward(self, hs, a):
        out = torch.concat(x=[hs, a], axis=-1)
        out = self.network(out)
        hidden = self.hs(out)
        reward = self.r(out)

        return hidden, reward


class Prediction_Model(nn.Module):
    def __init__(self, hidden_dim, action_dim):
        super().__init__()

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        network = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        ]
        self.network = nn.Sequential(*network)

        self.pi = nn.Linear(hidden_dim, self.action_dim)
        self.soft = nn.Softmax()

        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = self.network(x)
        p = self.pi(out)
        p = self.soft(p)

        v = self.v(out)
        return v, p


class MuZeroAgent(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.representation_model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.dynamics_model = Dynamics_Model(self.hidden_dim, self.action_dim)
        self.prediction_model = Prediction_Model(self.hidden_dim,
                                                 self.action_dim)

    def forward(self, s, a):
        s_0 = self.representation_model(s)
        s_1, r_1 = self.dynamics_model(s_0, a)
        value, p = self.prediction_model(s_1)

        return r_1, value, p


def choose_action(env, evaluate=False):
    values = []
    # mu.eval()
    for a in range(env.action_space.n):
        e = copy.deepcopy(env)
        o, r, d, _ = e.step(a)
        act = np.zeros(env.action_space.n)
        act[a] = 1
        state = torch.to_tensor(list(e.state), dtype='float32')
        action = torch.to_tensor(act, dtype='float32')
        # print(state,action)

        rew, v, pi = mu(state, action)
        v = v.numpy()[0]
        values.append(v)
    # mu.train()
    if evaluate:
        return np.argmax(values)
    else:
        for i in range(len(values)):
            if values[i] < 0:
                values[i] = 0
        s = sum(values)
        if s == 0:
            return np.random.choice(values)
        for i in range(len(values)):
            values[i] /= s
        # print(values)
        return np.random.choice(range(env.action_space.n), p=values)


def main():

    gamma = 0.997
    batch_size = 64  ##64
    evaluate = False
    scores = []
    avg_scores = []
    epochs = 2_000

    env = gym.make('CartPole-v0')
    hidden_dim = 128
    obs_dim = env.observation_space.shape[0]
    buffer = deque(maxlen=500)

    muzero_agent = MuZeroAgent(128, 2)

    optim = torch.optim.Adam(muzero_agent.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    for episode in tqdm(range(epochs)):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            a = choose_action(env, evaluate=evaluate)
            a_pi = np.zeros((env.action_space.n))
            a_pi[a] = 1

            obs_, r, done, _ = env.step(a)
            score += r
            buffer.append([obs, None, a_pi, r / 200])
            obs = obs_

        #print(f'score: {score}')
        scores.append(score)

        if len(scores) >= 100:
            avg_scores.append(np.mean(scores[-100:]))
        else:
            avg_scores.append(np.mean(scores))

        cnt = score
        for i in range(len(buffer)):
            if buffer[i][1] == None:
                buffer[i][1] = cnt / 200
                cnt -= 1
        assert (cnt == 0)

        if len(buffer) >= batch_size:
            batch = []
            indexes = np.random.choice(len(buffer), batch_size, replace=False)

            for i in range(batch_size):
                batch.append(buffer[indexes[i]])
            states = torch.to_tensor([transition[0] for transition in batch],
                                     dtype='float32')
            values = torch.to_tensor([transition[1] for transition in batch],
                                     dtype='float32')
            values = torch.reshape(values, [batch_size, -1])
            policies = torch.to_tensor([transition[2] for transition in batch],
                                       dtype='float32')
            rewards = torch.to_tensor([transition[3] for transition in batch],
                                      dtype='float32')
            rewards = torch.reshape(rewards, [batch_size, -1])
            for _ in range(2):
                # mu.train_on_batch([states, policies], [rewards, values, policies])
                rew, v, pi = mu(states, policies)
                # print("----rew---{}----v---{}----------pi---{}".format(rew, v, pi))
                # print("----rewards---{}----values---{}----------policies---{}".format(rewards, values, policies))

                policy_loss = -torch.mean(
                    torch.sum(policies * torch.log(pi), axis=1))
                mse1 = mse_loss(rew, rewards)
                mse2 = mse_loss(v, values)
                # print(mse1,mse2 ,policy_loss)

                loss = torch.add_n([policy_loss, mse1, mse2])
                # print(loss)
                loss.backward()
                optim.step()

        # 模型保存
        model_state_dict = muzero_agent.state_dict()
        torch.save(model_state_dict, 'mu.pt')
