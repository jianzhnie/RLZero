import logging
import os
import pprint
import threading
import time
import timeit
import traceback
from typing import Dict, List, Tuple

os.environ['OMP_NUM_THREADS'] = '1'  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from rlzero.algorithms.impala import vtrace


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages**2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction='none',
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def act(
    args,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    buffers: Dict[str, List[torch.Tensor]],
    initial_agent_state_buffers,
) -> None:
    try:
        logging.info('Actor %i started.', actor_index)

        gym_env = create_env(args)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_state = model.initial_state(batch_size=1)
        agent_output, unused_state = model(env_output, agent_state)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout.
            for t in range(args.unroll_length):
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state)

                env_output = env.step(agent_output['action'])

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

            full_queue.put(index)

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error('Exception in worker process %i', actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    args,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock(),
) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(args.batch_size)]
        timings.time('dequeue')
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    initial_agent_state = (torch.cat(ts, dim=1) for ts in zip(
        *[initial_agent_state_buffers[m] for m in indices]))
    timings.time('batch')
    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')
    batch = {
        k: t.to(device=args.device, non_blocking=True)
        for k, t in batch.items()
    }
    initial_agent_state = tuple(
        t.to(device=args.device, non_blocking=True)
        for t in initial_agent_state)
    timings.time('device')
    return batch, initial_agent_state


def learn(
        args,
        actor_model,
        model,
        batch,
        initial_agent_state,
        optimizer,
        scheduler,
        lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs, unused_state = model(batch, initial_agent_state)

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs['baseline'][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        if args.reward_clipping == 'abs_one':
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif args.reward_clipping == 'none':
            clipped_rewards = rewards

        discounts = (~batch['done']).float() * args.discounting

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs['policy_logits'],
            batch['action'],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = args.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = args.entropy_cost * compute_entropy_loss(
            learner_outputs['policy_logits'])

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch['episode_return'][batch['done']]
        stats = {
            'episode_returns': tuple(episode_returns.cpu().numpy()),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(rollout_length: int, num_buffers: int,
                   obs_shape: Tuple[int, ...], num_actions: int) -> Buffers:
    specs = dict(
        frame=dict(size=(rollout_length + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(rollout_length + 1, ), dtype=torch.float32),
        done=dict(size=(rollout_length + 1, ), dtype=torch.bool),
        episode_return=dict(size=(rollout_length + 1, ), dtype=torch.float32),
        episode_step=dict(size=(rollout_length + 1, ), dtype=torch.int32),
        policy_logits=dict(size=(rollout_length + 1, num_actions),
                           dtype=torch.float32),
        baseline=dict(size=(rollout_length + 1, ), dtype=torch.float32),
        last_action=dict(size=(rollout_length + 1, ), dtype=torch.int64),
        action=dict(size=(rollout_length + 1, ), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(args):  # pylint: disable=too-many-branches, too-many-statements
    if args.xpid is None:
        args.xpid = 'torchbeast-%s' % time.strftime('%Y%m%d-%H%M%S')
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' %
                           (args.savedir, args.xpid, 'model.tar')))

    if args.num_buffers is None:  # Set sensible default for num_buffers.
        args.num_buffers = max(2 * args.num_actors, args.batch_size)
    if args.num_actors >= args.num_buffers:
        raise ValueError('num_buffers should be larger than num_actors')
    if args.num_buffers < args.batch_size:
        raise ValueError('num_buffers should be larger than batch_size')

    T = args.unroll_length
    B = args.batch_size

    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        logging.info('Using CUDA.')
        args.device = torch.device('cuda')
    else:
        logging.info('Not using CUDA.')
        args.device = torch.device('cpu')

    env = create_env(args)

    model = Net(env.observation_space.shape, env.action_space.n, args.use_lstm)
    buffers = create_buffers(args, env.observation_space.shape,
                             model.num_actions)

    model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(args.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(args.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                args,
                i,
                free_queue,
                full_queue,
                model,
                buffers,
                initial_agent_state_buffers,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(env.observation_space.shape, env.action_space.n,
                        args.use_lstm).to(device=args.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        eps=args.epsilon,
        alpha=args.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, args.total_steps) / args.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
    ]
    logger.info('# Step\t%s', '\t'.join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        while step < args.total_steps:
            batch, agent_state = get_batch(
                args,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
            )
            stats = learn(args, model, learner_model, batch, agent_state,
                          optimizer, scheduler)
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                step += T * B

    for m in range(args.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(args.num_learner_threads):
        thread = threading.Thread(target=batch_and_learn,
                                  name='batch-and-learn-%d' % i,
                                  args=(i, ))
        thread.start()
        threads.append(thread)

    def checkpoint():
        if args.disable_checkpoint:
            return
        logging.info('Saving checkpoint to %s', checkpointpath)
        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'args': vars(args),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < args.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = ('Return per episode: %.1f. ' %
                               stats['mean_episode_return'])
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            logging.info(
                'Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s',
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info('Learning finished after %d steps.', step)
    finally:
        for _ in range(args.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
