"""
Remote client for napari-pyspim.

Manages SSH connection and MessagePack-based communication with the
remote computation server.
"""

import importlib.resources
import json
import os
import queue
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import msgpack
import numpy as np
import paramiko
from qtpy.QtCore import QObject, Signal, Slot, QThread


def encode_array(obj):
    """Encode numpy arrays for MessagePack serialization."""
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(0, msgpack.packb({
            'dtype': str(obj.dtype),
            'shape': obj.shape,
            'data': obj.tobytes(),
        }))
    raise TypeError(f"Cannot encode object of type {type(obj)}")


def decode_array(code, data, header):
    """Decode numpy arrays from MessagePack ext type."""
    if code == 0:
        meta = msgpack.unpackb(data, raw=False)
        arr = np.frombuffer(meta['data'], dtype=np.dtype(meta['dtype']))
        return arr.reshape(meta['shape'])
    return msgpack.ExtType(code, data)


def _read_exact(stream, n):
    """Read exactly n bytes from a stream."""
    data = b''
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _read_message(stdout):
    """Read a single length-prefixed MessagePack message from stdout.

    Returns the unpacked message dict, or None on EOF/connection error.
    """
    header = _read_exact(stdout, 8)
    if len(header) < 8:
        return None  # EOF
    length = int.from_bytes(header, 'big')
    data = _read_exact(stdout, length)
    if len(data) < length:
        return None  # EOF
    return msgpack.unpackb(data, ext_hook=decode_array, raw=False)


class _SenderThread(QThread):
    """Thread for sending messages to the remote server."""

    def __init__(self, stdin_stream):
        super().__init__()
        self._stdin = stdin_stream
        self._send_queue = queue.Queue()
        self._running = True

    def send(self, message):
        """Queue a message for sending."""
        packed = msgpack.packb(message, default=encode_array)
        length = len(packed)
        self._send_queue.put((length.to_bytes(8, 'big'), packed))

    def stop(self):
        """Signal the thread to stop."""
        self._running = False
        self._send_queue.put(None)  # Sentinel

    def run(self):
        """Send messages from the queue."""
        while self._running:
            try:
                item = self._send_queue.get(timeout=0.5)
                if item is None:
                    break
                header, data = item
                self._stdin.write(header)
                self._stdin.write(data)
                self._stdin.flush()
            except queue.Empty:
                continue
            except (BrokenPipeError, OSError):
                break


class _ReceiverThread(QThread):
    """Thread for receiving messages from the remote server."""

    message_received = Signal(dict)

    def __init__(self, stdout_stream):
        super().__init__()
        self._stdout = stdout_stream
        self._running = True

    def stop(self):
        """Signal the thread to stop."""
        self._running = False

    def run(self):
        """Read messages from stdout."""
        while self._running:
            try:
                message = _read_message(self._stdout)
                if message is None:
                    break  # EOF
                self.message_received.emit(message)
            except (OSError, EOFError, msgpack.ExtraData):
                break


class RemoteClient(QObject):
    """Manages SSH connection and communication with remote compute server.

    Signals:
        connected: Emitted when SSH connection is established.
        disconnected: Emitted when SSH connection is closed.
        error: Emitted when an error occurs (str message).
        progress: Emitted with progress updates from remote server (str).
    """

    connected = Signal()
    disconnected = Signal()
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ssh = None  # paramiko.SSHClient
        self._transport = None  # paramiko.Transport
        self._sftp = None  # paramiko.SFTPClient
        self._sender = None  # _SenderThread
        self._receiver = None  # _ReceiverThread
        self._pending_callbacks = {}  # request_id -> callback
        self._request_id_counter = 0
        self._lock = threading.Lock()
        self._connected = False

        # Connection settings
        self.host = None
        self.port = 22
        self.username = None
        self.key_path = None
        self.remote_venv = None

        # Config file path
        self._config_dir = Path.home() / '.pyspim'
        self._config_path = self._config_dir / 'remote_config.json'

    @property
    def is_connected(self):
        """Return True if SSH connection is active."""
        return self._connected

    @property
    def sftp_client(self):
        """Return the SFTP client for file operations."""
        return self._sftp

    # --- Connection Management ---

    def load_config(self) -> dict:
        """Load saved connection settings from config file."""
        if self._config_path.exists():
            try:
                with open(self._config_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def save_config(self, config: dict):
        """Save connection settings to config file (excludes password/passphrase)."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        # Never save password or passphrase
        safe_config = {k: v for k, v in config.items()
                       if k not in ('password', 'key_passphrase')}
        with open(self._config_path, 'w') as f:
            json.dump(safe_config, f, indent=2)

    def connect(
        self,
        host: str,
        port: int = 22,
        username: str = None,
        auth_method: str = 'key',  # 'password' or 'key'
        password: str = None,
        key_path: str = None,
        key_passphrase: str = None,
        remote_venv: str = None,
    ) -> bool:
        """Establish SSH connection and start remote Python session.

        Returns True if connection was successful.
        """
        try:
            self.host = host
            self.port = port
            self.username = username
            self.key_path = key_path
            self.remote_venv = remote_venv

            self.progress.emit(f"Connecting to {host}:{port}...")

            # Create SSH client
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            # Connect
            connect_kwargs = {
                'hostname': host,
                'port': port,
                'username': username,
                'look_for_keys': False,
                'allow_agent': False,
            }
            if auth_method == 'password':
                connect_kwargs['password'] = password
            else:
                if key_path:
                    connect_kwargs['key_filename'] = key_path
                if key_passphrase:
                    connect_kwargs['passphrase'] = key_passphrase

            self._ssh.connect(**connect_kwargs)
            self._transport = self._ssh.get_transport()

            # Start SFTP
            self._sftp = paramiko.SFTPClient.from_transport(self._transport)

            # Upload and start remote server
            self._start_remote_server()

            if self._sender is not None and self._receiver is not None:
                self._receiver.message_received.connect(self._on_message_received)
                self._sender.start()
                self._receiver.start()

            self._connected = True
            self.connected.emit()
            return True

        except Exception as e:
            self.error.emit(f"Connection failed: {e}")
            self.disconnect_ssh()
            return False

    def _start_remote_server(self):
        """Upload remote server script and start it on the remote server.

        Reads the server's "ready" message to verify the process started
        successfully before returning.  Raises RuntimeError if the server
        fails to start or does not send a valid ready message.
        """
        self.progress.emit("Uploading server script...")

        # Get the remote server script content
        try:
            server_source = importlib.resources.files('napari_pyspim').joinpath('_remote_server.py').read_text()
        except Exception:
            # Fallback: read from package path
            server_path = os.path.join(os.path.dirname(__file__), '_remote_server.py')
            with open(server_path, 'r') as f:
                server_source = f.read()

        # Upload to temp location on remote server
        remote_script_path = '/tmp/_pyspim_remote_server.py'
        with self._sftp.open(remote_script_path, 'w') as remote_file:
            remote_file.write(server_source)
            remote_file.flush()

        # Build command to activate venv and run server
        if self.remote_venv:
            python_bin = os.path.join(self.remote_venv, 'bin', 'python')
            cmd = f'{python_bin} {remote_script_path}'
        else:
            cmd = f'python3 {remote_script_path}'

        self.progress.emit("Starting remote server process...")

        # Open session and get stdin/stdout
        channel = self._transport.open_session()
        channel.get_pty()
        channel.exec_command(cmd)
        self._stdin = channel.makefile('wb', 0)
        self._stdout = channel.makefile('rb', 0)

        # --- Health check: read the server's "ready" message ---
        # The remote server sends a ready message immediately on startup.
        # Read it synchronously to verify the process is alive before
        # handing the streams over to the sender/receiver threads.
        self.progress.emit("Waiting for server ready message...")

        ready_message = None
        deadline = time.monotonic() + 30.0

        while time.monotonic() < deadline:
            if channel.exit_status_ready():
                # Process already exited -- try to collect any remaining output
                # for diagnostics (PTY merges stdout and stderr).
                exit_code = channel.recv_exit_status()
                remaining_parts = []
                _read_deadline = time.monotonic() + 3.0
                while time.monotonic() < _read_deadline and channel.recv_ready():
                    chunk = channel.recv(4096)
                    if not chunk:
                        break
                    remaining_parts.append(chunk.decode(errors='replace'))
                remaining_text = ''.join(remaining_parts).strip()
                raise RuntimeError(
                    f"Remote server process exited immediately (exit code {exit_code}). "
                    f"Command: {cmd}"
                    + (f"\nOutput: {remaining_text}" if remaining_text else "")
                )
            if channel.recv_ready():
                ready_message = _read_message(self._stdout)
                break
            time.sleep(0.05)

        if ready_message is None:
            raise RuntimeError(
                "Remote server did not send a ready message within 30 seconds. "
                "The server process may have failed to start. "
                f"Command used: {cmd}"
            )

        if not ready_message.get('success', False):
            error_msg = ready_message.get('error', 'Unknown error')
            raise RuntimeError(f"Remote server reported error on startup: {error_msg}")

        self.progress.emit("Remote server ready.")

        # Start sender/receiver threads
        self._sender = _SenderThread(self._stdin)
        self._receiver = _ReceiverThread(self._stdout)

    @Slot(dict)
    def _on_message_received(self, message):
        """Handle incoming message from remote server."""
        request_id = message.get('request_id')
        success = message.get('success', False)
        result = message.get('result')
        error = message.get('error')
        progress = message.get('progress')

        if progress:
            self.progress.emit(progress)

        if request_id is not None:
            with self._lock:
                callback = self._pending_callbacks.pop(request_id, None)
            if callback:
                callback(success, result, error)
        else:
            # Unsolicited message (e.g., ready, progress)
            if not success and error:
                self.error.emit(error)

    def disconnect_ssh(self):
        """Terminate remote session and close SSH connection."""
        self._connected = False

        # Cancel all pending callbacks
        with self._lock:
            for callback in self._pending_callbacks.values():
                callback(False, None, 'Connection disconnected')
            self._pending_callbacks.clear()

        # Stop threads
        if self._sender:
            self._sender.stop()
            self._sender.wait(3000)
        if self._receiver:
            self._receiver.stop()
            self._receiver.wait(3000)

        # Clean up remote server script
        try:
            if self._sftp:
                try:
                    self._sftp.remove('/tmp/_pyspim_remote_server.py')
                except IOError:
                    pass
                self._sftp.close()
        except Exception:
            pass

        # Close SSH connection
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass

        self._ssh = None
        self._transport = None
        self._sftp = None
        self._sender = None
        self._receiver = None

        self.disconnected.emit()

    # --- Command Execution ---

    def _next_request_id(self):
        """Generate the next request ID."""
        with self._lock:
            rid = self._request_id_counter
            self._request_id_counter += 1
            return rid

    def send_command(self, command: str, params: dict,
                     callback: Callable[[bool, Optional[dict], Optional[str]], None]):
        """Send a command to the remote server.

        Args:
            command: Command name (e.g., 'ping', 'load_deskew').
            params: Command parameters as a dict.
            callback: Function called with (success, result, error) when response arrives.
        """
        if not self._connected:
            callback(False, None, 'Not connected to remote server')
            return

        # Validate sender/receiver threads are still alive
        if self._sender is None or not self._sender.isRunning():
            callback(False, None, 'Sender thread is not running - connection may be broken')
            return
        if self._receiver is None or not self._receiver.isRunning():
            callback(False, None, 'Receiver thread is not running - connection may be broken')
            return

        request_id = self._next_request_id()
        with self._lock:
            self._pending_callbacks[request_id] = callback

        message = {
            'request_id': request_id,
            'command': command,
            'params': params,
        }
        self.progress.emit(f"Sending '{command}' command (id={request_id})...")
        self._sender.send(message)

    def send_command_blocking(self, command: str, params: dict,
                              timeout: float = 300.0) -> dict:
        """Send a command and wait for response.

        Args:
            command: Command name.
            params: Command parameters.
            timeout: Maximum time to wait in seconds.

        Returns:
            Result dict from remote server.

        Raises:
            RuntimeError: If command fails or times out.
        """
        result_queue = queue.Queue()

        def callback(success, result, error):
            result_queue.put((success, result, error))

        self.send_command(command, params, callback)
        self.progress.emit(f"Waiting for '{command}' response (timeout={timeout}s)...")

        try:
            success, result, error = result_queue.get(timeout=timeout)
        except queue.Empty:
            # Diagnose: check if threads are still alive
            sender_ok = self._sender is not None and self._sender.isRunning()
            receiver_ok = self._receiver is not None and self._receiver.isRunning()
            raise RuntimeError(
                f"Command '{command}' timed out after {timeout}s. "
                f"Sender thread alive: {sender_ok}, Receiver thread alive: {receiver_ok}"
            )

        if not success:
            raise RuntimeError(f"Command '{command}' failed: {error}")

        return result

    # --- SFTP Helpers ---

    def list_directory(self, remote_path: str) -> list:
        """List directory contents via remote server command."""
        result = self.send_command_blocking('list_directory', {'path': remote_path})
        return result.get('entries', [])

    def check_path(self, remote_path: str) -> dict:
        """Check if a path exists on the remote server."""
        return self.send_command_blocking('check_path', {'path': remote_path})

    # --- Remote Zarr Helpers ---

    def open_remote_zarr(self, zarr_path: str):
        """Open a remote zarr file via fsspec for chunked reading.

        Returns a zarr array that streams chunks over SSH on demand.
        """
        import fsspec
        import zarr

        # Create SSH filesystem
        fs_kwargs = {
            'host': self.host,
            'port': self.port,
            'username': self.username,
            'known_hosts': None,
        }
        if self.key_path:
            fs_kwargs['key_filename'] = self.key_path

        fs = fsspec.filesystem('ssh', **fs_kwargs)

        # Create zarr store from SSH filesystem
        store = zarr.storage.FSStore(zarr_path, fs=fs)
        return zarr.open(store, mode='r')

    def save_zarr(self, temp_path: str, permanent_path: str):
        """Move zarr archive from temp to permanent location on remote server."""
        return self.send_command_blocking('save_zarr', {
            'temp_path': temp_path,
            'permanent_path': permanent_path,
        })

    def cleanup_zarr(self, temp_path: str):
        """Remove a temp zarr directory on the remote server."""
        return self.send_command_blocking('cleanup_zarr', {
            'temp_path': temp_path,
        })

    # --- Ping / Health Check ---

    def ping(self) -> dict:
        """Send a ping to check server health."""
        return self.send_command_blocking('ping', {}, timeout=10.0)
