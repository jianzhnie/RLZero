from .player import Player


class HumanPlayer(Player):

    def __init__(self, player_id=0, player_name=''):
        Player.__init__(self, player_id, player_name)
        self.can_click = True  # can click the board

    def play(self, board):
        """play based on human input."""
        try:
            location = input('Your move: ')
            if isinstance(location, str):
                location = [int(n, 10)
                            for n in location.split(',')]  # for python3
            move = board.loc2move(location)
        except Exception as e:
            print(e)
            move = -1
        if move == -1 or move not in board.availables:
            print('invalid move')
            move = self.play(board)
        return move

    def play_with_toool(self, board, **kwargs):
        tool = kwargs['tool']
        while not tool.flag:  # block
            pass
        location = tool.getmove()  # [x,y]
        move = board.loc2move(location)
        if move == -1 or move not in board.availables:
            print('invalid move')
            move = self.play(board)
        return move

    def __str__(self):
        return 'HumanPlayer {}{}'.format(self.get_player_id(),
                                         self.get_player_name())
