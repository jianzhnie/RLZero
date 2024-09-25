# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

# Worker and gather module

import copy
import functools
import multiprocessing as mp
import pickle
import queue
import random
import threading
import time
from collections import deque
from socket import gethostname
from typing import Any, Dict, List, Optional, Tuple

from .connection import (QueueCommunicator, accept_socket_connections,
                         connect_socket_connection,
                         open_multiprocessing_connections, send_recv)
from .environment import make_env, prepare_env
from .evaluation import Evaluator
from .generation import Generator
from .model import ModelWrapper, RandomModel


class Worker:
    """A class responsible for managing worker processes that interact with the
    environment. It generates episodes or evaluations based on assigned roles.

    Attributes:
        args (Dict): Worker configuration and environment arguments.
        conn (Any): Connection object to communicate with other processes or the server.
        wid (int): Worker ID to distinguish different workers.
    """

    def __init__(self, args: Dict[str, Any], conn: Any, wid: int):
        """Initializes the worker with environment, generator, and evaluator
        instances.

        Args:
            args: Configuration and arguments for the worker.
            conn: Connection for worker-server communication.
            wid: Worker ID to differentiate between workers.
        """
        print(f'Opened worker {wid}')
        self.worker_id = wid
        self.args = args
        self.conn = conn
        self.latest_model = (-1, None)

        # Initialize environment, generator, and evaluator
        self.env = make_env({**args['env'], 'id': wid})
        self.generator = Generator(self.env, self.args)
        self.evaluator = Evaluator(self.env, self.args)

        random.seed(args['seed'] + wid)

    def __del__(self):
        """Cleanup upon destruction of the worker."""
        print(f'Closed worker {self.worker_id}')

    def _gather_models(
            self, model_ids: List[int]) -> Dict[int, Optional[ModelWrapper]]:
        """Fetches models from the server or uses cached models in the worker.

        Args:
            model_ids: List of model IDs to retrieve from the server or cache.

        Returns:
            Dictionary of models mapped to model IDs.
        """
        model_pool = {}
        for model_id in model_ids:
            if model_id not in model_pool:
                if model_id < 0:
                    model_pool[model_id] = None
                elif model_id == self.latest_model[0]:
                    # Use the latest cached model
                    model_pool[model_id] = self.latest_model[1]
                else:
                    # Request the model from the server
                    model = pickle.loads(
                        send_recv(self.conn, ('model', model_id)))
                    if model_id == 0:
                        # Use random model for model_id 0
                        self.env.reset()
                        obs = self.env.observation(self.env.players()[0])
                        model = RandomModel(model, obs)
                    model_pool[model_id] = ModelWrapper(model)
                    # Update the latest model cache
                    if model_id > self.latest_model[0]:
                        self.latest_model = model_id, model_pool[model_id]
        return model_pool

    def run(self):
        """Main execution loop for the worker to receive and handle tasks."""
        while True:
            args = send_recv(self.conn, ('args', None))
            if args is None:
                break

            role = args['role']
            models = {}

            if 'model_id' in args:
                model_ids = list(args['model_id'].values())
                model_pool = self._gather_models(model_ids)

                # Create a dictionary of models for players
                for player, model_id in args['model_id'].items():
                    models[player] = model_pool[model_id]

            # Execute generation or evaluation based on the role
            if role == 'g':
                episode = self.generator.execute(models, args)
                send_recv(self.conn, ('episode', episode))
            elif role == 'e':
                result = self.evaluator.execute(models, args)
                send_recv(self.conn, ('result', result))


def make_worker_args(args: Dict[str, Any], n_ga: int, gaid: int, base_wid: int,
                     wid: int, conn: Any) -> Tuple[Dict[str, Any], Any, int]:
    """Constructs the arguments for initializing a worker.

    Args:
        args: General worker arguments.
        n_ga: Number of gather processes.
        gaid: Gather ID.
        base_wid: Base worker ID.
        wid: Worker ID.
        conn: Connection object for worker-server communication.

    Returns:
        A tuple of worker arguments, connection object, and worker ID.
    """
    return args, conn, base_wid + wid * n_ga + gaid


def open_worker(args: Dict[str, Any], conn: Any, wid: int):
    """Function to open a worker process and start its run loop.

    Args:
        args: Arguments for the worker.
        conn: Connection object.
        wid: Worker ID.
    """
    worker = Worker(args, conn, wid)
    worker.run()


class Gather(QueueCommunicator):
    """A gather class responsible for handling multiple worker processes,
    buffering data, and communicating with a server for task assignment and
    result gathering."""

    def __init__(self, args: Dict[str, Any], conn: Any, gaid: int):
        """Initializes the gather process, opens worker connections, and sets
        up buffers.

        Args:
            args: Gather-specific arguments and worker configurations.
            conn: Connection to the server.
            gaid: Gather ID.
        """
        print(f'Started gather {gaid}')
        super().__init__()
        self.gather_id = gaid
        self.server_conn = conn
        self.args_queue = deque()
        self.data_map = {'model': {}}
        self.result_send_map = {}
        self.result_send_cnt = 0

        n_pro, n_ga = args['worker']['num_parallel'], args['worker'][
            'num_gathers']
        num_workers_per_gather = (n_pro // n_ga) + int(gaid < n_pro % n_ga)
        base_wid = args['worker'].get('base_worker_id', 0)

        worker_conns = open_multiprocessing_connections(
            num_workers_per_gather,
            open_worker,
            functools.partial(make_worker_args, args, n_ga, gaid, base_wid),
        )

        for conn in worker_conns:
            self.add_connection(conn)

        self.buffer_length = 1 + len(worker_conns) // 4

    def __del__(self):
        """Cleanup upon destruction of the gather process."""
        print(f'Finished gather {self.gather_id}')

    def run(self):
        """Main loop to receive commands from workers and handle tasks."""
        while self.connection_count() > 0:
            try:
                conn, (command, args) = self.recv(timeout=0.3)
            except queue.Empty:
                continue

            if command == 'args':
                if not self.args_queue:
                    self.server_conn.send(
                        (command, [None] * self.buffer_length))
                    self.args_queue += self.server_conn.recv()

                next_args = self.args_queue.popleft()
                self.send(conn, next_args)

            elif command in self.data_map:
                data_id = args
                if data_id not in self.data_map[command]:
                    self.server_conn.send((command, args))
                    self.data_map[command][data_id] = self.server_conn.recv()
                self.send(conn, self.data_map[command][data_id])

            else:
                self.send(conn, None)
                if command not in self.result_send_map:
                    self.result_send_map[command] = []
                self.result_send_map[command].append(args)
                self.result_send_cnt += 1

                if self.result_send_cnt >= self.buffer_length:
                    for command, args_list in self.result_send_map.items():
                        self.server_conn.send((command, args_list))
                        self.server_conn.recv()
                    self.result_send_map.clear()
                    self.result_send_cnt = 0


def gather_loop(args: Dict[str, Any], conn: Any, gaid: int):
    """Entry point for starting a gather process."""
    gather = Gather(args, conn, gaid)
    gather.run()


class WorkerCluster(QueueCommunicator):
    """Manages the worker cluster and opens local connections for workers."""

    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.args = args

    def run(self):
        """Main loop to initialize and manage gathers and workers."""
        if 'num_gathers' not in self.args['worker']:
            self.args['worker']['num_gathers'] = (
                1 + max(0, self.args['worker']['num_parallel'] - 1) // 16)

        for i in range(self.args['worker']['num_gathers']):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=gather_loop, args=(self.args, conn1, i)).start()
            conn1.close()
            self.add_connection(conn0)


class WorkerServer(QueueCommunicator):
    """Handles incoming worker connections and assigns tasks."""

    def __init__(self, args: Dict[str, Any]):
        super().__init__()
        self.args = args
        self.total_worker_count = 0

    def run(self):
        """Start the server threads to accept connections and manage
        workers."""

        def entry_server(port: int):
            print(f'Started entry server {port}')
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                worker_args = conn.recv()
                print(f'Accepted connection from {worker_args["address"]}')
                worker_args['base_worker_id'] = self.total_worker_count
                self.total_worker_count += worker_args['num_parallel']
                args = copy.deepcopy(self.args)
                args['worker'] = worker_args
                conn.send(args)
                conn.close()

        def worker_server(port: int):
            print(f'Started worker server {port}')
            conn_acceptor = accept_socket_connections(port=port)
            while True:
                conn = next(conn_acceptor)
                self.add_connection(conn)

        threading.Thread(target=entry_server, args=(9999, ),
                         daemon=True).start()
        threading.Thread(target=worker_server, args=(9998, ),
                         daemon=True).start()


def entry(worker_args: Dict[str, Any]) -> Dict[str, Any]:
    """Establishes a connection to the server and retrieves initial
    arguments."""
    conn = connect_socket_connection(worker_args['server_address'], 9999)
    conn.send(worker_args)
    args = conn.recv()
    conn.close()
    return args


class RemoteWorkerCluster:
    """Manages a remote cluster of workers that connect to a central server."""

    def __init__(self, args: Dict[str, Any]):
        args['address'] = gethostname()
        if 'num_gathers' not in args:
            args['num_gathers'] = 1 + max(0, args['num_parallel'] - 1) // 16

        self.args = args

    def run(self):
        """Run the worker cluster and continuously manage worker processes."""
        args = entry(self.args)
        print(args)
        prepare_env(args['env'])

        processes = []
        try:
            for i in range(self.args['num_gathers']):
                conn = connect_socket_connection(self.args['server_address'],
                                                 9998)
                p = mp.Process(target=gather_loop, args=(args, conn, i))
                p.start()
                conn.close()
                processes.append(p)

            while True:
                time.sleep(100)

        finally:
            for p in processes:
                p.terminate()


def worker_main(args: Dict[str, Any], argv: List[str]):
    """Main entry point for starting worker processes."""
    worker_args = args['worker_args']
    if len(argv) >= 1:
        worker_args['num_parallel'] = int(argv[0])

    worker = RemoteWorkerCluster(args=worker_args)
    worker.run()
