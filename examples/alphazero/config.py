import torch


class Config:

    def __init__(self) -> None:
        self.game = 'Gamoku'
        self.num_rows = 8
        self.num_cols = 8
        self.path = './work_dirs'
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.train_batch_size = 2**10
        self.replay_buffer_size = 2**16
        self.replay_buffer_reuse = 3
        self.max_steps = 0
        self.checkpoint_freq = 100

        self.actors = 1
        self.evaluators = 1
        self.evaluation_window = 100
        self.eval_levels = 7

        self.uct_c = 2
        self.max_simulations = 300
        self.policy_alpha = 1
        self.policy_epsilon = 0.25
        self.temperature = 1
        self.temperature_drop = 10

        self.observation_shape = None
        self.output_size = None

        self.quiet = True


class AlphaZeroConfig:

    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game
        # params of the board and the game
        self.game = 'Gamoku'
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4

        # Training
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.work_dir = 'work_dirs/'

        # training params
        self.episode_size = 10000  # total episode for the model to train
        self.epochs = 5  # num of train_steps for each update
        self.batch_size = 128  # mini-batch size for training
        self.learning_rate = 2e-3
        self.temperature = 1.0  # the temperature param
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        self.c_puct = 5  # MCTS PUCT method param
        self.n_playout = 400  # num of simulations for each move
        self.kl_targ = 0.02
        self.check_freq = 10  # freq for test
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        self.n_games = 10

        # Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
