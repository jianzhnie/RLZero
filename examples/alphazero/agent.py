import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import PolicyValueNet


class AlphaZeroAgent(object):

    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cpu') -> None:
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.device = device
        self.policy_value_net = PolicyValueNet(num_rows, num_cols)
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

        current_state = torch.from_numpy(current_state).float().to(self.device)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.items()
        return act_probs, value

    def learn(self, observation, legals_mask, policy_targets, value_targets):
        """perform a training step."""
        # train mode
        self.policy_value_net.train()
        policy_logits, value_out = self.policy_value_net(observation)
        # policy_softmax = F.softmax(policy_logits)
        policy_log_probs = F.log_softmax(policy_logits)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value_out.view(-1), value_targets)
        # policy loss
        policy_loss = -F.cross_entropy(policy_logits, policy_targets)
        # total loss
        loss = value_loss + policy_loss

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(policy_log_probs) * policy_log_probs, 1))
        return loss.item(), policy_loss.item(), value_loss.item(
        ), entropy.item()

    def predict(self, observation, legals_mask):
        self.policy_value_net.eval()  # eval mode
        observation = torch.tensor(observation, device=self.device)
        legals_mask = torch.tensor(legals_mask, device=self.device)
        with torch.no_grad():
            policy_logits, value_out = self.policy_value_net(observation)

        policy_softmax = F.softmax(policy_logits)
        return value_out, policy_softmax

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
