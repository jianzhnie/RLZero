import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
import gymnasium as gym
import random

# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



# Actor进程
def actor_process(actor_id, env, q_network, data_queue):
    state, _ = env.reset()
    buffer = []
    def get_action(state):
        
        # 选择动作
        with torch.no_grad():
            if state.ndim == 1:
                # Expand to have batch_size = 1
                state = np.expand_dims(state, axis=0)
            
            state = torch.tensor(state, dtype=torch.float)
            q_values = q_network(state)
            action = q_values.argmax().item()

        return action 

    while True:
        action = get_action(state)
        # 与环境交互
        next_state, reward, terminal, truncated, _ = env.step(action)
        done = terminal or truncated
        buffer.append((state, action, reward, next_state, done))
        state = next_state

        # 如果缓冲区满了，发送数据给Learner
        if len(buffer) > 100:
            data_queue.put(buffer)
            buffer = []

        if done:
            state, _ = env.reset()

# Learner进程
def learner_process(data_queue, param_queue, q_network, target_network, optimizer, replay_buffer, gamma=0.99, batch_size=32):
    
    global_step = 0
    total_steps = 10000

    while global_step < total_steps:
        # 从Actor接收数据
        if not data_queue.empty():
            data = data_queue.get()
            actor_step = len(data)
            global_step += actor_step
            for experience in data:
                replay_buffer.push(experience)

        # 如果回放缓冲区足够大，进行训练
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            # 将数据转换为np.array
            states = np.stack(states, axis=0)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.stack(next_states, axis=0)
            dones = np.array(dones)

            # 将数据转换为torch张量
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
            rewards = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)

            q_values = q_network(states).gather(1, actions)
            next_q_values = target_network(next_states).max(1, keepdim=True)[0]
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, expected_q_values.detach())
            print("global_step: %s, loss: %s", (global_step, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 定期更新目标网络
            if np.random.rand() < 0.01:
                target_network.load_state_dict(q_network.state_dict())

            # 将更新后的Q网络参数发送给Actor
            param_queue.put(q_network.state_dict())

# 主函数
def main():
    state_dim = 4  # 状态维度
    action_dim = 2  # 动作维度
    num_actors = 4  # Actor数量
    replay_buffer_capacity = 10000  # 回放缓冲区容量

    q_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    data_queue = mp.Queue()
    param_queue = mp.Queue()
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    actors = []
    for i in range(num_actors):
        env = gym.make('CartPole-v1')  # 使用OpenAI Gym环境
        actor = mp.Process(target=actor_process, args=(i, env, q_network, data_queue))
        actors.append(actor)
        actor.start()

    learner = mp.Process(target=learner_process, args=(data_queue, param_queue, q_network, target_network, optimizer, replay_buffer))
    learner.start()

    for actor in actors:
        actor.join()
    learner.join()

if __name__ == "__main__":
    main()