import numpy as np

from .alphazero_mcts import AlphaZeroMCTS
from .player import Player


class AlphaZeroPlayer(Player):
    """AI player based on MCTS."""

    def __init__(
        self,
        policy_value_fn,
        n_playout: int = 2000,
        c_puct: float = 5,
        is_selfplay: bool = False,
        add_noise: bool = False,
    ) -> None:
        self.mcts = AlphaZeroMCTS(
            policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=is_selfplay,
        )
        self.is_selfplay = is_selfplay
        self._add_noise = is_selfplay if add_noise is None else add_noise

    def reset_player(self):
        """reset, reconstructing the MCTS Tree for next simulation."""
        self.mcts.update_with_move(-1)

    def get_action(
        self,
        game_env,
        temperature: float = 1e-3,
        return_prob: bool = False,
    ):
        sensible_moves = game_env.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(game_env.width * game_env.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.simulate(game_env, temperature)
            move_probs[list(acts)] = probs
            move = np.random.choice(acts, p=probs)
            if self.is_selfplay:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                # reset the root node
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print('WARNING: the board is full')

    def __str__(self):
        return 'AlphaZeroPlayer {}{}'.format(self.get_player_id()(),
                                             self.get_player_name())
