import io
import multiprocessing as mp
import multiprocessing.connection as connection
import pickle
import queue
import socket
import struct
import threading
from typing import Any, Callable, Generator, List, Optional, Tuple


def send_recv(conn: connection.Connection, sdata: Any) -> Any:
    """Send data through the connection and wait to receive a response.

    Args:
        conn (connection.Connection): Connection object for communication.
        sdata (Any): Data to be sent.

    Returns:
        Any: Received data.
    """
    conn.send(sdata)
    return conn.recv()


class PickledConnection:
    """A class to handle sending and receiving pickled (serialized) data over a
    connection."""

    def __init__(self, conn: socket.socket):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Close the connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def fileno(self) -> int:
        """Return the file descriptor of the connection."""
        return self.conn.fileno()

    def _recv(self, size: int) -> io.BytesIO:
        """Receive a specified amount of data."""
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            if len(chunk) == 0:
                raise ConnectionResetError('Connection reset by peer')
            size -= len(chunk)
            buf.write(chunk)
        return buf

    def recv(self) -> Any:
        """Receive a complete message.

        The first 4 bytes represent the size of the message.
        """
        buf = self._recv(4)
        (size, ) = struct.unpack('!i', buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf: bytes) -> None:
        """Send a buffer of data."""
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg: Any) -> None:
        """Send a serialized message, prepending the size as a 4-byte
        header."""
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack('!i', n)
        chunks = [header + buf] if n <= 16384 else [header, buf]
        for chunk in chunks:
            self._send(chunk)


def open_socket_connection(port: int, reuse: bool = False) -> socket.socket:
    """Open a socket connection on a specified port.

    Args:
        port (int): Port number to bind the socket to.
        reuse (bool): Whether to reuse the address (default is False).

    Returns:
        socket.socket: A bound socket ready to listen for connections.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1 if reuse else 0)
    sock.bind(('', port))
    return sock


def accept_socket_connection(
        sock: socket.socket) -> Optional[PickledConnection]:
    """Accept a socket connection and return a PickledConnection wrapper.

    Args:
        sock (socket.socket): The listening socket.

    Returns:
        Optional[PickledConnection]: A wrapper for the accepted connection, or None if timed out.
    """
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


def listen_socket_connections(n: int,
                              port: int) -> List[Optional[PickledConnection]]:
    """Listen for a specified number of socket connections.

    Args:
        n (int): Number of connections to accept.
        port (int): Port to listen on.

    Returns:
        List[Optional[PickledConnection]]: A list of connections.
    """
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def connect_socket_connection(host: str, port: int) -> PickledConnection:
    """Connect to a remote host and port, returning a PickledConnection.

    Args:
        host (str): Host address.
        port (int): Port number.

    Returns:
        PickledConnection: A connection object that handles pickled communication.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        raise ConnectionError(f'Failed to connect to {host}:{port}')
    return PickledConnection(sock)


def accept_socket_connections(
        port: int,
        timeout: Optional[float] = None,
        maxsize: int = 1024
) -> Generator[Optional[PickledConnection], None, None]:
    """Accept multiple socket connections.

    Args:
        port (int): Port number to listen on.
        timeout (Optional[float]): Timeout for accepting connections.
        maxsize (int): Maximum number of connections to accept.

    Yields:
        Optional[PickledConnection]: Generator of PickledConnections or None.
    """
    sock = open_socket_connection(port)
    sock.listen(maxsize)
    sock.settimeout(timeout)
    cnt = 0
    while cnt < maxsize:
        conn = accept_socket_connection(sock)
        if conn is not None:
            cnt += 1
        yield conn


def open_multiprocessing_connections(
    num_process: int,
    target: Callable,
    args_func: Callable[[int, connection.Connection], Tuple],
) -> List[connection.Connection]:
    """Open multiprocessing connections and start workers.

    Args:
        num_process (int): Number of worker processes.
        target (Callable): Target function for each process.
        args_func (Callable[[int, connection.Connection], Tuple]): Function returning arguments for each process.

    Returns:
        List[connection.Connection]: List of parent connections.
    """
    s_conns, g_conns = [], []
    for _ in range(num_process):
        conn0, conn1 = mp.Pipe(duplex=True)
        s_conns.append(conn0)
        g_conns.append(conn1)

    for i, conn in enumerate(g_conns):
        mp.Process(target=target, args=args_func(i, conn)).start()
        conn.close()

    return s_conns


class MultiProcessJobExecutor:
    """Executor for managing and dispatching jobs to multiple worker
    processes."""

    def __init__(
        self,
        func: Callable,
        send_generator: Generator,
        num_workers: int,
        postprocess: Optional[Callable] = None,
    ):
        """Initialize the executor with worker processes.

        Args:
            func (Callable): Worker process function.
            send_generator (Generator): Generator yielding data to be sent.
            num_workers (int): Number of worker processes.
            postprocess (Optional[Callable]): Function to process received data before returning.
        """
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.conns = []
        self.waiting_conns = queue.Queue()
        self.output_queue = queue.Queue(maxsize=8)

        for i in range(num_workers):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=func, args=(conn1, i), daemon=True).start()
            conn1.close()
            self.conns.append(conn0)
            self.waiting_conns.put(conn0)

    def recv(self) -> Any:
        """Receive the next result from the output queue."""
        return self.output_queue.get()

    def start(self) -> None:
        """Start the sender and receiver threads."""
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._receiver, daemon=True).start()

    def _sender(self) -> None:
        """Continuously send data to available workers."""
        print('start sender')
        while True:
            data = next(self.send_generator)
            conn = self.waiting_conns.get()
            conn.send(data)
        print('finished sender')

    def _receiver(self) -> None:
        """Continuously receive data from workers and enqueue the results."""
        print('start receiver')
        while True:
            conns = connection.wait(self.conns)
            for conn in conns:
                data = conn.recv()
                self.waiting_conns.put(conn)
                if self.postprocess is not None:
                    data = self.postprocess(data)
                self.output_queue.put(data)
        print('finished receiver')


class QueueCommunicator:
    """Class for handling asynchronous communication using queues."""

    def __init__(self, conns: List[connection.Connection] = []):
        """Initialize the communicator with a set of connections.

        Args:
            conns (List[connection.Connection]): Initial set of connections.
        """
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns = set(conns)
        threading.Thread(target=self._send_thread, daemon=True).start()
        threading.Thread(target=self._recv_thread, daemon=True).start()

    def connection_count(self) -> int:
        """Return the number of active connections."""
        return len(self.conns)

    def recv(self, timeout: Optional[float] = None) -> Any:
        """Receive data from the input queue."""
        return self.input_queue.get(timeout=timeout)

    def send(self, conn: connection.Connection, send_data: Any) -> None:
        """Send data via a specified connection."""
        self.output_queue.put((conn, send_data))

    def add_connection(self, conn: connection.Connection) -> None:
        """Add a new connection."""
        self.conns.add(conn)

    def disconnect(self, conn: connection.Connection) -> None:
        """Disconnect and remove a connection."""
        print('disconnected')
        self.conns.discard(conn)

    def _send_thread(self) -> None:
        """Thread to handle sending data through connections."""
        while True:
            conn, send_data = self.output_queue.get()
            try:
                conn.send(send_data)
            except (TimeoutError, ConnectionResetError, BrokenPipeError):
                self.disconnect(conn)

    def _recv_thread(self) -> None:
        """Thread to handle receiving data through connections."""
        while True:
            conns = connection.wait(self.conns, timeout=0.3)
            for conn in conns:
                try:
                    recv_data = conn.recv()
                except (TimeoutError, ConnectionResetError, EOFError):
                    self.disconnect(conn)
                    continue
                self.input_queue.put((conn, recv_data))
