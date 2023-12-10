from typing import Any

import numpy as np

from .alphazero_mcts import AlphaZeroMCTS


class MCTSPlayer(object):
    """AI player based on MCTS."""

    def __init__(
        self,
        policy_value_function,
        c_puct: float = 5,
        n_playout: int = 2000,
        is_selfplay: bool = False,
    ) -> None:
        self.mcts = AlphaZeroMCTS(
            policy_value_function,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=is_selfplay,
        )
        self.is_selfplay = is_selfplay

    def set_player_ind(self, player_id: int) -> None:
        """set player index."""
        self.player = player_id

    def reset_player(self):
        """reset player."""
        self.mcts.update_with_move(-1)

    def get_action(
        self,
        game_env,
        temperature: float = 1e-3,
        return_prob: bool = False,
    ) -> tuple[Any, np.NDArray]:
        sensible_moves = game_env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(game_env.width * game_env.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(game_env, temperature)
            move_probs[list(acts)] = probs
            if self.is_selfplay:
                move = np.random.choice(acts, p=probs)
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'MCTS-AlphaZero {}'.format(self.player)
