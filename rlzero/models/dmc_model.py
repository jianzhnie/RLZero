from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn


class DMCNet(nn.Module):
    """A fully connected neural network that takes a state and an action as
    input and outputs a scalar value. This can be used in model-based
    reinforcement learning where the network predicts values or rewards based
    on state-action pairs.

    Args:
        state_shape (Tuple[int]): The shape of the input state (observation).
        action_shape (Tuple[int]): The shape of the input action.
        mlp_layers (List[int]): List of integers defining the size of each hidden layer.
                                Defaults to [512, 512, 512, 512, 512].
    """

    def __init__(
        self,
        state_shape: Tuple[int],
        action_shape: Tuple[int],
        mlp_layers: List[int] = [512, 512, 512, 512, 512],
    ) -> None:
        super(DMCNet, self).__init__()

        # Compute the input dimension as the sum of the state and action shapes
        input_dim = np.prod(state_shape) + np.prod(action_shape)

        # Define the dimensions of each fully connected layer
        layer_dims = [input_dim] + mlp_layers

        # Create a list to hold the layers of the network
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers.append(nn.ReLU())  # ReLU activation after each Linear layer

        # Final output layer which predicts a single scalar value
        layers.append(nn.Linear(layer_dims[-1], 1))

        # Wrap all layers in a Sequential container
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            obs (torch.Tensor): A batch of observations (states), shape (batch_size, *state_shape).
            actions (torch.Tensor): A batch of actions, shape (batch_size, *action_shape).

        Returns:
            torch.Tensor: A batch of scalar values (predictions) for each state-action pair.
        """
        # Flatten the observations and actions (if multi-dimensional)
        obs = torch.flatten(obs,
                            1)  # Flatten starting from the second dimension
        # Flatten starting from the second dimension
        actions = torch.flatten(actions, 1)

        # Concatenate the flattened observations and actions along the feature dimension
        x = torch.cat((obs, actions), dim=1)

        # Pass the concatenated input through the fully connected layers
        values = self.fc_layers(
            x).flatten()  # Flatten the output to a 1D tensor

        return values


class DMCAgent(nn.Module):
    """Deep Model-based Control Agent using a neural network (DMCNet) to
    predict Q-values for state-action pairs in a reinforcement learning
    setting. The agent selects actions based on epsilon-greedy policy during
    training and greedy policy during evaluation.

    Args:
        state_shape (Tuple[int]): The shape of the input state (observation).
        action_shape (Tuple[int]): The shape of the input action.
        mlp_layers (List[int], optional): List of integers defining the size of each hidden layer.
                                          Defaults to [512, 512, 512, 512, 512].
        exp_epsilon (float, optional): The probability of choosing a random action (exploration).
                                       Defaults to 0.01.
        device (str, optional): Device to be used for computations, either 'cpu' or a GPU index
                                (e.g., '0' for 'cuda:0'). Defaults to '0'.
    """

    def __init__(
        self,
        state_shape: Tuple[int],
        action_shape: Tuple[int],
        mlp_layers: list = [512, 512, 512, 512, 512],
        exp_epsilon: float = 0.01,
        device: Union[str, int] = '0',
    ) -> None:
        self.use_raw = False
        self.device = f'cuda:{device}' if device != 'cpu' else 'cpu'
        self.net = DMCNet(state_shape, action_shape,
                          mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def step(self, state: Dict[str, Any]) -> Any:
        """Take a step in the environment by selecting an action based on the
        current state.

        Args:
            state (Dict): A dictionary containing the current state and legal actions.

        Returns:
            Any: The chosen action.
        """
        action_keys, values = self.predict(state)

        # Epsilon-greedy action selection
        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            action = np.random.choice(action_keys)
        else:
            action_idx = np.argmax(values)
            action = action_keys[action_idx]

        return action

    def eval_step(self, state: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Take an evaluation step (without exploration), selecting the best
        action based on the current state.

        Args:
            state (Dict): A dictionary containing the current state and legal actions.

        Returns:
            Tuple[Any, Dict]: The chosen action and additional information including predicted values.
        """
        action_keys, values = self.predict(state)

        # Greedy action selection
        action_idx = np.argmax(values)
        action = action_keys[action_idx]

        info = {
            'values': {
                state['raw_legal_actions'][i]: float(values[i])
                for i in range(len(action_keys))
            }
        }

        return action, info

    def share_memory(self):
        """Make the network shareable across processes."""
        self.net.share_memory()

    def eval(self):
        """Set the network to evaluation mode."""
        self.net.eval()

    def parameters(self):
        """Return the parameters of the network."""
        return self.net.parameters()

    def predict(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict Q-values for all legal actions given the current state.

        Args:
            state (Dict): A dictionary containing 'obs' (current observation) and 'legal_actions'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The action keys and their corresponding predicted Q-values.
        """
        obs = state['obs'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_keys = np.array(list(legal_actions.keys()))
        action_values = list(legal_actions.values())

        # One-hot encoding for actions without features
        for i in range(len(action_values)):
            if action_values[i] is None:
                action_values[i] = np.zeros(self.action_shape[0])
                action_values[i][action_keys[i]] = 1
        action_values = np.array(action_values, dtype=np.float32)

        # Repeat observations to match the number of actions
        obs = np.repeat(obs[np.newaxis, :], len(action_keys), axis=0)

        # Predict Q-values
        values = self.net(
            torch.from_numpy(obs).to(self.device),
            torch.from_numpy(action_values).to(self.device),
        )

        return action_keys, values.cpu().detach().numpy()

    def forward(self, obs: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            obs (torch.Tensor): The batch of observations.
            actions (torch.Tensor): The batch of actions.

        Returns:
            torch.Tensor: The predicted Q-values.
        """
        return self.net.forward(obs, actions)

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the state dictionary for the network."""
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        """Return the state dictionary of the network."""
        return self.net.state_dict()

    def set_device(self, device: str):
        """Set the device for the agent's network.

        Args:
            device (str): The device identifier, e.g., 'cpu' or 'cuda:0'.
        """
        self.device = device


class DMCModel:
    """DMCModel manages a set of DMCAgents, each corresponding to a player in
    the environment. Each agent interacts with its own state and action space,
    and the model facilitates collective operations across agents, such as
    sharing memory or switching to evaluation mode.

    Args:
        state_shape (List[Tuple[int]]): A list of shapes, where each shape corresponds to the
                                        observation (state) shape for each agent.
        action_shape (List[Tuple[int]]): A list of shapes, where each shape corresponds to the
                                         action space shape for each agent.
        mlp_layers (List[int], optional): The structure of the multi-layer perceptron for each agent.
                                          Defaults to [512, 512, 512, 512, 512].
        exp_epsilon (float, optional): Exploration rate for epsilon-greedy action selection.
                                       Defaults to 0.01.
        device (int or str, optional): The device to run the model on, either a GPU index (int) or
                                       "cpu". Defaults to 0 (i.e., 'cuda:0').
    """

    def __init__(
        self,
        state_shape: List[Tuple[int]],
        action_shape: List[Tuple[int]],
        mlp_layers: List[int] = [512, 512, 512, 512, 512],
        exp_epsilon: float = 0.01,
        device: str = '0',
    ) -> None:
        # Initialize agents for each player, each with their respective state and action shapes
        self.agents: List[DMCAgent] = [
            DMCAgent(
                state_shape[player_id],
                action_shape[player_id],
                mlp_layers,
                exp_epsilon,
                device,
            ) for player_id in range(len(state_shape))
        ]

    def share_memory(self) -> None:
        """Share memory for all agents to enable multiprocessing."""
        for agent in self.agents:
            agent.share_memory()

    def eval(self) -> None:
        """Set all agents to evaluation mode."""
        for agent in self.agents:
            agent.eval()

    def parameters(self, index: int):
        """Get the parameters of the agent at the specified index.

        Args:
            index (int): The index of the agent to retrieve parameters for.

        Returns:
            Iterator over agent parameters.
        """
        return self.agents[index].parameters()

    def get_agent(self, index: int) -> DMCAgent:
        """Retrieve a specific agent by index.

        Args:
            index (int): The index of the agent.

        Returns:
            DMCAgent: The agent corresponding to the given index.
        """
        return self.agents[index]

    def get_agents(self) -> List[DMCAgent]:
        """Retrieve the list of all agents.

        Returns:
            List[DMCAgent]: The list of all agents.
        """
        return self.agents
