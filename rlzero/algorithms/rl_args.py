import argparse
from dataclasses import dataclass, field


@dataclass
class RLArguments:
    """Settings for the DouZero PyTorch DouDizhu AI."""

    # General settings
    project: str = field(default='rlzero',
                         metadata={'help': 'Project name (default: rlzero)'})
    algo_name: str = field(default='dmc',
                           metadata={'help': 'Algorithm name (default: dqn)'})
    objective: str = field(
        default='adp',
        metadata={
            'help': 'Use ADP or WP as reward (default: ADP)',
            'choices': ['adp', 'wp', 'logadp'],
        },
    )
    env_id: str = field(
        default='CartPole-v1',
        metadata={'help': 'Environment ID (default: CartPole-v1)'})
    # Training settings
    use_cuda: bool = field(default=True,
                           metadata={'help': 'Use CUDA (default: True)'})
    actor_device_cpu: bool = field(
        default=True, metadata={'help': 'Use CPU as actor device'})
    gpu_devices: str = field(
        default='0', metadata={'help': 'Which GPUs to be used for training'})
    num_actor_devices: int = field(
        default=1,
        metadata={'help': 'The number of devices used for simulation'})
    num_actors: int = field(
        default=1,
        metadata={'help': 'The number of actors for each simulation device'})
    training_device: str = field(
        default='0',
        metadata={
            'help':
            'The index of the GPU used for training models. `cpu` means using CPU'
        },
    )
    load_model: bool = field(default=False,
                             metadata={'help': 'Load an existing model'})
    disable_checkpoint: bool = field(
        default=False, metadata={'help': 'Disable saving checkpoint'})
    savedir: str = field(
        default='douzero_checkpoints',
        metadata={'help': 'Root dir where experiment data will be saved'},
    )
    save_interval: int = field(
        default=30,
        metadata={
            'help': 'Time interval (in minutes) at which to save the model'
        },
    )
    # Hyperparameters
    total_steps: int = field(
        default=100_000_000_000,
        metadata={'help': 'Total environment steps to train for'},
    )
    epsilon_greedy: float = field(
        default=0.01, metadata={'help': 'The probability for exploration'})
    batch_size: int = field(default=32,
                            metadata={'help': 'Learner batch size'})
    rollout_length: int = field(
        default=100, metadata={'help': 'The rollout length (time dimension)'})
    num_buffers: int = field(
        default=50, metadata={'help': 'Number of shared-memory buffers'})
    num_threads: int = field(default=4,
                             metadata={'help': 'Number learner threads'})
    max_grad_norm: float = field(default=40.0,
                                 metadata={'help': 'Max norm of gradients'})

    # Optimizer settings
    learning_rate: float = field(default=0.0001,
                                 metadata={'help': 'Learning rate'})
    alpha: float = field(default=0.99,
                         metadata={'help': 'RMSProp smoothing constant'})
    momentum: float = field(default=0.0, metadata={'help': 'RMSProp momentum'})
    epsilon: float = field(default=1e-5, metadata={'help': 'RMSProp epsilon'})


def parse_args() -> RLArguments:
    """Parses command-line arguments using argparse and returns an RLArguments
    instance.

    Returns:
        RLArguments: Populated RLArguments dataclass instance with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='DouZero: PyTorch DouDizhu AI')

    # Automatically populate arguments based on the RLArguments dataclass
    for field_name, field_info in RLArguments.__dataclass_fields__.items():
        help_msg = field_info.metadata.get('help', '')
        if 'choices' in field_info.metadata:
            parser.add_argument(
                f'--{field_name}',
                type=type(field_info.default),
                default=field_info.default,
                choices=field_info.metadata['choices'],
                help=help_msg,
            )
        else:
            parser.add_argument(
                f'--{field_name}',
                type=type(field_info.default),
                default=field_info.default,
                help=help_msg,
            )

    args = parser.parse_args()
    return RLArguments(**vars(args))


if __name__ == '__main__':
    args = parse_args()
    print(args)
