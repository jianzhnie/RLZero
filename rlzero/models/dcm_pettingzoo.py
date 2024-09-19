from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from pettingzoo import AECEnv

from rlzero.utils.pettingzoo_utils import wrap_state

from .dmc_model import DMCAgent


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
