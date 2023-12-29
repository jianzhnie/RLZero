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


class HumanPlayer(Player):

    def __init__(self, player_id=0, player_name=''):
        super().__init__(player_id, player_name)
        self.can_click = True  # can click the board

    def get_action(self, game_env, **kwargs):
        """play based on human input."""
        try:
            location = input('Your move: ')
            if isinstance(location, str):
                location = [int(n, 10)
                            for n in location.split(',')]  # for python3
            move = game_env.location_to_move(location)
        except Exception as e:
            print(e)
            move = -1
        if move == -1 or move not in game_env.leagel_actions():
            print('invalid move')
            move = self.get_action(game_env)
        return move

    def __str__(self):
        return 'HumanPlayer, id: {}, name {}.'.format(self.get_player_id(),
                                                      self.get_player_name())
