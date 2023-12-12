"""Base Player Class extract the Play Abstract Method for different player to
override."""


class Player(object):

    def __init__(self, player_id=0, player_name='') -> None:
        self.player_id = player_id
        self.player_name = player_name
        self.can_click = False

    def set_player_id(self, player_id):
        self.player_id = player_id

    def get_player_id(self):
        return self.player_id

    def get_player_name(self):
        return self.player_name

    def reset_player(self):
        """reset, reconstructing the MCTS Tree for next simulation."""
        raise NotImplementedError

    # abstract
    def get_action(self, game_env, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return 'player'
