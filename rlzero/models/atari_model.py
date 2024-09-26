from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtariNet(nn.Module):
    """A neural network for playing Atari games using convolutional layers
    followed by optional LSTM layers for sequential decision making."""

    def __init__(
        self,
        observation_shape: Tuple[int, int, int],
        num_actions: int,
        use_lstm: bool = False,
    ):
        """Initializes the AtariNet model.

        Args:
            observation_shape (Tuple[int, int, int]): Shape of the input observations (C, H, W).
            num_actions (int): Number of possible actions to take.
            use_lstm (bool): Whether to use LSTM for the model's core.
        """
        super(AtariNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions

        # Feature extraction layers
        self.conv1 = nn.Conv2d(
            in_channels=self.observation_shape[0],
            out_channels=32,
            kernel_size=8,
            stride=4,
        )
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        # Fully connected layer
        self.fc = nn.Linear(3136, 512)

        # Determine core output size
        core_output_size = self.fc.out_features + num_actions + 1

        self.use_lstm = use_lstm
        if use_lstm:
            self.core = nn.LSTM(core_output_size,
                                core_output_size,
                                num_layers=2)

        # Policy and baseline outputs
        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self,
                      batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the initial hidden state for the LSTM.

        Args:
            batch_size (int): The size of the input batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial states for LSTM.
        """
        if not self.use_lstm:
            return tuple()
        return tuple(
            torch.zeros(self.core.num_layers, batch_size,
                        self.core.hidden_size) for _ in range(2))

    def forward(
        self,
        inputs: Dict[str, Any],
        core_state: Tuple[torch.Tensor, torch.Tensor] = ()
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the network.

        Args:
            inputs (Dict[str, Any]): Input data containing frames, last action, reward, and done flags.
            core_state (Tuple[torch.Tensor, torch.Tensor]): Hidden state for LSTM.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: Outputs and updated core state.
        """
        x = inputs['frame']  # [T, B, C, H, W]
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch
        x = x.float() / 255.0  # Normalize input

        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(T * B, -1)  # Flatten for fully connected layer
        x = F.relu(self.fc(x))

        # Prepare input for LSTM or fully connected layers
        one_hot_last_action = F.one_hot(inputs['last_action'].view(T * B),
                                        self.num_actions).float()
        clipped_reward = torch.clamp(inputs['reward'], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward, one_hot_last_action],
                               dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs['done']).float()
            for input_seq, no_do in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ends
                no_do = no_do.view(1, -1, 1)
                core_state = tuple(no_do * s for s in core_state)
                output, core_state = self.core(input_seq.unsqueeze(0),
                                               core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        # Calculate policy logits and baseline
        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        # Action selection
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                       num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        # Reshape outputs
        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits,
                    baseline=baseline,
                    action=action), core_state
