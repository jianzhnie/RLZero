from .player import Player


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

    def get_action_tool(self, board, **kwargs):
        tool = kwargs['tool']
        while not tool.flag:  # block
            pass
        location = tool.getmove()  # [x,y]
        move = board.loc2move(location)
        if move == -1 or move not in board.leagel_actions():
            print('invalid move')
            move = self.get_action_tool(board)
        return move

    def __str__(self):
        return 'HumanPlayer, id: {}, name {}.'.format(self.get_player_id(),
                                                      self.get_player_name())
