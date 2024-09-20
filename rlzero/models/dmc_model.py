from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from pettingzoo import AECEnv
from torch import nn

from rlzero.utils.pettingzoo_utils import wrap_state


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
        # Flatten starting from the second dimension
        obs = torch.flatten(obs, 1)
        # Flatten starting from the second dimension
        actions = torch.flatten(actions, 1)

        # Concatenate the flattened observations and actions along the feature dimension
        x = torch.cat((obs, actions), dim=1)

        # Pass the concatenated input through the fully connected layers
        # Flatten the output to a 1D tensor
        values = self.fc_layers(x).flatten()

        return values


class DMCAgent(object):
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
        values = self.net.forward(
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
        state_shapes (List[Tuple[int]]): A list of shapes, where each shape corresponds to the
                                        observation (state) shape for each agent.
        action_shapes (List[Tuple[int]]): A list of shapes, where each shape corresponds to the
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
        state_shapes: List[Tuple[int]],
        action_shapes: List[Tuple[int]],
        mlp_layers: List[int] = [512, 512, 512, 512, 512],
        exp_epsilon: float = 0.01,
        device: Union[str, int] = '0',
    ) -> None:
        # Initialize agents for each player, each with their respective state and action shapes
        self.agents: List[DMCAgent] = [
            DMCAgent(
                state_shapes[player_id],
                action_shapes[player_id],
                mlp_layers,
                exp_epsilon,
                device,
            ) for player_id in range(len(state_shapes))
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


class DMCAgentPettingZoo(DMCAgent):
    """A wrapper for the DMCAgent class that adapts it to work with
    PettingZoo's multi-agent framework.

    This class ensures that the environment states are wrapped appropriately
    before interacting with the DMCAgent methods.
    """

    def step(self, state: Dict[str, Any]) -> Any:
        """Override the step function to wrap the input state.

        Args:
            state: The current state of the environment.

        Returns:
            The agent's action after processing the wrapped state.
        """
        return super().step(wrap_state(state))

    def eval_step(self, state: Dict[str, Any]) -> Any:
        """Override the eval_step function to wrap the input state for
        evaluation.

        Args:
            state: The current state of the environment.

        Returns:
            The evaluated action of the agent.
        """
        return super().eval_step(wrap_state(state))

    def feed(self, ts: Tuple[Any, Any, float, Any, bool]) -> None:
        """Override the feed function to wrap the states in the transition
        tuple (ts).

        Args:
            ts: A tuple containing (state, action, reward, next_state, done) information.
        """
        state, action, reward, next_state, done = ts
        state = wrap_state(state)
        next_state = wrap_state(next_state)
        wrapped_ts = (state, action, reward, next_state, done)
        super().feed(wrapped_ts)


class DMCModelPettingZoo:
    """A model for handling multiple agents in a PettingZoo environment.

    This class maintains a dictionary of DMCAgentPettingZoo instances, one for
    each agent in the environment.
    """

    def __init__(
        self,
        env: AECEnv,
        mlp_layers: List[int] = [512, 512, 512, 512, 512],
        exp_epsilon: float = 0.01,
        device: str = '0',
    ):
        """Initializes the DMCModelPettingZoo with multiple agents based on the
        environment.

        Args:
            env: The PettingZoo environment instance.
            mlp_layers: The layers for the multi-layer perceptron (MLP) in the agent model.
            exp_epsilon: The exploration epsilon value for the agents.
            device: The device to be used by the agents (e.g., "0" for GPU, or "cpu").
        """
        self.agents: OrderedDict[str, DMCAgentPettingZoo] = OrderedDict()

        # Create an agent for each agent in the environment
        for agent_name in env.agents:
            agent = DMCAgentPettingZoo(
                env.observation_space(agent_name)['observation'].shape,
                (env.action_space(agent_name).n, ),
                mlp_layers,
                exp_epsilon,
                device,
            )
            self.agents[agent_name] = agent

    def share_memory(self) -> None:
        """Share memory between agents (for multiprocessing)."""
        for agent in self.agents.values():
            agent.share_memory()

    def eval(self) -> None:
        """Set all agents to evaluation mode (disables exploration)."""
        for agent in self.agents.values():
            agent.eval()

    def parameters(self, index: int) -> Any:
        """Get the parameters of the agent at the specified index.

        Args:
            index: The index of the agent in the OrderedDict.

        Returns:
            The parameters of the specified agent.
        """
        return list(self.agents.values())[index].parameters()

    def get_agent(self, index: int) -> DMCAgentPettingZoo:
        """Retrieve a specific agent by index.

        Args:
            index: The index of the agent.

        Returns:
            The DMCAgentPettingZoo instance corresponding to the specified index.
        """
        return list(self.agents.values())[index]

    def get_agents(self) -> List[DMCAgentPettingZoo]:
        """Retrieve a list of all agents in the model.

        Returns:
            A list of all DMCAgentPettingZoo instances.
        """
        return list(self.agents.values())
