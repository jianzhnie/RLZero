import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .gomoku_env import GomokuEnv
from .policy_value_net import PolicyValueNet


class AlphaZeroAgent(object):

    def __init__(
        self,
        board_width: int,
        board_height: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = 'cpu',
    ) -> None:
        self.board_width = board_width
        self.board_height = board_height
        self.policy_value_net = PolicyValueNet(board_width, board_height)
        self.policy_value_net.to(device)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.device = device

    def policy_value_fn(self, game_env: GomokuEnv):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = game_env.leagel_actions()
        current_state = game_env.current_state().reshape(
            -1, 4, self.board_width, self.board_height)
        current_state = np.ascontiguousarray(current_state)
        current_state = torch.from_numpy(current_state).float().to(self.device)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.item()
        return act_probs, value

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        value = value.detach().cpu().numpy()
        return act_probs, value

    def learn(self, state_batch, mcts_probs, target_vs):
        """perform a training step."""
        # train mode
        self.policy_value_net.train()
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
        target_batch = torch.FloatTensor(target_vs).to(self.device)

        log_act_probs, value = self.policy_value_net(state_batch)

        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), target_batch)
        # policy loss
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        # total loss
        loss = value_loss + policy_loss

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1))
        return loss.item(), entropy.item()

    def predict(self, state_batch):
        self.policy_value_net.eval()  # eval mode
        state_batch = torch.FloatTensor(state_batch).to(self.device)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)

        act_probs = np.exp(log_act_probs.detach().cpu().numpy())
        value = value.detach().cpu().numpy()
        return act_probs, value

    def save_model(
        self,
        save_dir: str,
        model_name: str = 'model.th',
        opt_name: str = 'optimizer.th',
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_path = os.path.join(save_dir, model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        torch.save(self.policy_value_net.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        print('save model successfully!')

    def restore(
        self,
        save_dir: str,
        model_name: str = 'model.th',
        opt_name: str = 'optimizer.th',
    ):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_path = os.path.join(save_dir, model_name)
        optimizer_path = os.path.join(save_dir, opt_name)
        self.policy_value_net.load_state_dict(torch.load(model_path))
        self.optimizer.load_state_dict(torch.load(optimizer_path))
        print('restore model successfully!')
