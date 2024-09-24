# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# Episode generation module

import bz2
import pickle
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .util import softmax


class Generator:
    """A class responsible for generating episodes of environment interactions.

    Attributes:
        env: The environment in which the interactions occur.
        args: Additional arguments or settings for episode generation.
    """

    def __init__(self, env: Any, args: Dict[str, Any]):
        """Initialize the generator with environment and arguments.

        Args:
            env: The environment instance for the game or interaction.
            args: Additional settings for controlling episode generation.
        """
        self.env = env
        self.args = args

    def generate(
        self, models: Dict[str, Any], args: Dict[str, Any]
    ) -> Optional[Dict[str, Union[Dict[str, Any], List[bytes], int]]]:
        """Generate a single episode by interacting with the environment using
        the provided models.

        Args:
            models: Dictionary of models corresponding to players in the environment.
            args: Arguments controlling the generation process (e.g., player selection, observation settings).

        Returns:
            A dictionary containing compressed moments of the episode, step count, and outcomes, or None if an error occurs.
        """
        # Stores all moments in the episode
        moments = []

        # Store hidden states for each player
        hidden = {
            player: models[player].init_hidden()
            for player in self.env.players()
        }

        # Reset the environment, handle errors
        err = self.env.reset()
        if err:
            return None

        # Loop through each step until the environment reaches a terminal state
        while not self.env.terminal():
            # Prepare a dictionary to capture the moment for each player
            moment_keys = [
                'observation',
                'selected_prob',
                'action_mask',
                'action',
                'value',
                'reward',
                'return',
            ]
            moment = {
                key: {p: None
                      for p in self.env.players()}
                for key in moment_keys
            }

            # Identify players and observers for the current turn
            turn_players = self.env.turns()
            observers = self.env.observers()

            for player in self.env.players():
                # Skip players that are not involved in this turn
                if player not in turn_players + observers:
                    continue
                # Skip players who do not need observations
                if (player not in turn_players and player in args['player']
                        and not self.args.get('observation', True)):
                    continue

                # Get the player's observation from the environment
                obs = self.env.observation(player)
                model = models[player]

                # Run the model's inference to get action probabilities, value estimate, etc.
                outputs = model.inference(obs, hidden[player])
                hidden[player] = outputs.get('hidden', None)
                value = outputs.get('value', None)

                moment['observation'][player] = obs
                moment['value'][player] = value

                if player in turn_players:
                    # Process action probabilities and select an action
                    policy_logits = outputs['policy']
                    legal_actions = self.env.legal_actions(player)

                    action_mask = (np.ones_like(policy_logits) * 1e32
                                   )  # Mask all actions by default
                    action_mask[legal_actions] = 0  # Allow only legal actions

                    # Calculate softmax of the masked logits to get action probabilities
                    action_probs = softmax(policy_logits - action_mask)

                    # Choose an action based on the probabilities
                    action = random.choices(
                        legal_actions, weights=action_probs[legal_actions])[0]

                    # Store the action information in the moment
                    moment['selected_prob'][player] = action_probs[action]
                    moment['action_mask'][player] = action_mask
                    moment['action'][player] = action

            # Apply the chosen actions to the environment
            err = self.env.step(moment['action'])
            if err:
                return None

            # Collect the reward for the current state
            rewards = self.env.reward()
            for player in self.env.players():
                moment['reward'][player] = rewards.get(player, None)

            # Add the turn information and store the moment
            moment['turn'] = turn_players
            moments.append(moment)

        if len(moments) < 1:
            return None

        # Compute returns (discounted future rewards) for each player
        for player in self.env.players():
            ret = 0
            for i, m in reversed(list(enumerate(moments))):
                ret = (m['reward'][player] or 0) + self.args['gamma'] * ret
                moments[i]['return'][player] = ret

        # Compress the episode moments into chunks
        episode = {
            'args':
            args,
            'steps':
            len(moments),
            'outcome':
            self.env.outcome(),
            'moment': [
                bz2.compress(
                    pickle.dumps(moments[i:i + self.args['compress_steps']]))
                for i in range(0, len(moments), self.args['compress_steps'])
            ],
        }

        return episode

    def execute(
        self, models: Dict[str, Any], args: Dict[str, Any]
    ) -> Optional[Dict[str, Union[Dict[str, Any], List[bytes], int]]]:
        """Generate and execute an episode, printing an error message if
        generation fails.

        Args:
            models: Dictionary of models corresponding to players.
            args: Additional arguments for controlling generation.

        Returns:
            The generated episode or None if generation fails.
        """
        episode = self.generate(models, args)
        if episode is None:
            print('None episode in generation!')
        return episode
