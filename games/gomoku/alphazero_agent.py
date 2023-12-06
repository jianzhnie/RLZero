import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import PolicyValueNet
from torch.autograd import Variable


class AlphaZeroAgent(object):

    def __init__(self,
                 board_width: int,
                 board_height: int,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu') -> None:
        self.board_width = board_width
        self.board_height = board_height

        self.device = device
        self.policy_value_net = PolicyValueNet(board_width, board_height)
        self.policy_value_net.to(device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))

        # current_state = torch.from_numpy(current_state).float().to(self.device)
        # log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = self.policy_value_net(
            Variable(torch.from_numpy(current_state).float()))

        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def learn(self, state_batch, mcts_probs, target_vs):
        """perform a training step."""
        # train mode
        self.policy_value_net.train()

        state_batch = np.array(state_batch)
        mcts_probs = np.array(mcts_probs)
        target_batch = np.array(target_vs)

        state_batch = Variable(torch.FloatTensor(state_batch))
        mcts_probs = Variable(torch.FloatTensor(mcts_probs))
        target_batch = Variable(torch.FloatTensor(target_batch))

        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), target_batch)
        # policy loss
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        # total loss
        loss = value_loss + policy_loss

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        return loss.item(), policy_loss.item(), value_loss.item(
        ), entropy.item()

    def predict(self, state_batch):
        self.policy_value_net.eval()  # eval mode
        state_batch = np.array(state_batch)
        state_batch = Variable(torch.FloatTensor(state_batch))

        with torch.no_grad():
            log_pi, value = self.policy_value_net(state_batch)

        act_probs = np.exp(log_pi.data.numpy())
        return act_probs, value.data.numpy()

    def save(self,
             save_dir: str,
             model_name: str = 'model.th',
             opt_name: str = 'optimizer.th'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_path = os.path.join(save_dir, model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.policy_value_net.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def restore(self,
                save_dir: str,
                model_name: str = 'model.th',
                opt_name: str = 'optimizer.th'):

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_path = os.path.join(save_dir, model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.policy_value_net.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
