"""Remote client for napari-pyspim.

Manages the SSH connection and communication with the remote server process
using length-prefixed MessagePack over SSH stdin/stdout.
"""

from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

import msgpack
import numpy as np
import paramiko
from qtpy.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Serialization helpers  (mirror of _remote_server.py)
# ---------------------------------------------------------------------------

def _encode_array(obj: Any) -> dict:
    """Encode a numpy array as a MessagePack-serializable dict."""
    if isinstance(obj, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    raise TypeError(f"Cannot encode object of type {type(obj)}")


def _decode_array(obj: dict) -> "np.ndarray | dict":
    """Decode a dict produced by ``_encode_array`` back to a numpy array."""
    if isinstance(obj, dict) and obj.get("__numpy__"):
        return np.frombuffer(
            obj["data"], dtype=np.dtype(obj["dtype"])
        ).reshape(obj["shape"])
    return obj


# ---------------------------------------------------------------------------
# RemoteClient
# ---------------------------------------------------------------------------

class RemoteClient(QObject):
    """Manages SSH connection and communication with remote compute server.

    Signals
    -------
    connected
        Emitted when the remote server is ready.
    disconnected
        Emitted when the connection is closed.
    error : str
        Emitted when an error occurs.
    progress : str, int
        Emitted for progress updates from the server (message, percentage).
    command_response : dict
        Emitted when a command response arrives from the server.
        The dict contains keys: request_id, success, result, error.
    """

    connected = Signal()
    disconnected = Signal()
    error = Signal(str)
    progress = Signal(str, int)
    command_response = Signal(dict)

    def __init__(self, parent=None):  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._ssh: paramiko.SSHClient | None = None
        self._channel: paramiko.Channel | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self._sender_thread: threading.Thread | None = None
        self._receiver_thread: threading.Thread | None = None
        self._message_queue: queue.Queue[dict | None] = queue.Queue()
        self._pending_callbacks: dict[int, Callable] = {}
        self._pending_progress: dict[int, Callable] = {}
        self._request_id: int = 0
        self._lock = threading.Lock()
        self._connected = False
        self._server_capabilities: dict = {}
        # Connection params for potential reuse
        self._host: str | None = None
        self._port: int | None = None
        self._username: str | None = None
        self._key_path: str | None = None
        self._remote_script_path: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(
        self,
        host: str,
        port: int,
        username: str,
        auth_method: str,
        password: str | None = None,
        key_path: str | None = None,
        key_passphrase: str | None = None,
        remote_venv: str | None = None,
    ) -> bool:
        """Establish SSH connection and start remote Python session.

        Parameters
        ----------
        host : str
            Remote server hostname or IP.
        port : int
            SSH port (default 22).
        username : str
            SSH username.
        auth_method : str
            ``"password"`` or ``"key"``.
        password : str, optional
            SSH password (required when auth_method="password").
        key_path : str, optional
            Path to local SSH private key (required when auth_method="key").
        key_passphrase : str, optional
            Passphrase for encrypted private key.
        remote_venv : str, optional
            Path to the pyspim virtualenv on the remote server.

        Returns
        -------
        bool
            True if connection succeeded.
        """
        try:
            self._host = host
            self._port = port
            self._username = username
            self._key_path = key_path

            # --- SSH connection ---
            self._ssh = paramiko.SSHClient()
            self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs: dict[str, Any] = {
                "hostname": host,
                "port": port,
                "username": username,
                "allow_agent": True,
                "look_for_keys": False,  # Only use explicitly provided key
                "timeout": 30,
            }

            if auth_method == "password":
                connect_kwargs["password"] = password
            elif auth_method == "key":
                connect_kwargs["key_filename"] = key_path
                if key_passphrase:
                    connect_kwargs["passphrase"] = key_passphrase

            self._ssh.connect(**connect_kwargs)

            # --- SFTP ---
            self._sftp = self._ssh.open_sftp()

            # --- Upload server script ---
            local_script = os.path.join(
                os.path.dirname(__file__), "_remote_server.py"
            )
            self._remote_script_path = (
                f"/tmp/pyspim_remote_server_{os.getpid()}.py"
            )
            self._sftp.put(local_script, self._remote_script_path)

            # --- Start remote Python process ---
            # Prefer the venv's python binary directly (more reliable than
            # sourcing activate which can fail in non-interactive shells).
            if remote_venv:
                python_bin = os.path.join(remote_venv, "bin", "python")
                # Derive log path: <pyspim_root>/logs/pyspim_server_<pid>.log
                # (pyspim_root = parent of the venv directory)
                pyspim_root = os.path.dirname(remote_venv)
                log_dir = os.path.join(pyspim_root, "logs")
                self._remote_log_path = os.path.join(
                    log_dir, f"pyspim_server_{os.getpid()}.log"
                )
                # Ensure the logs directory exists on the remote host
                try:
                    self._sftp.mkdir(log_dir)
                except IOError:
                    pass  # Directory already exists
                command = (
                    f"PYSPIM_LOG_PATH={self._remote_log_path} "
                    f"{python_bin} {self._remote_script_path} "
                    f"2>{self._remote_log_path}"
                )
            else:
                self._remote_log_path = None
                command = f"python {self._remote_script_path}"

            logger.info("[CONNECT] Remote command: %s", command)
            self._channel = self._ssh.get_transport().open_session()
            self._channel.exec_command(command)

            # --- Start sender thread ---
            self._sender_thread = threading.Thread(
                target=self._sender_loop, daemon=True
            )
            self._sender_thread.start()

            # --- Wait for ready message (before starting receiver loop) ---
            # Deliberately start _receiver_loop AFTER receiving the ready
            # message to avoid a race where two threads read from the same
            # SSH channel concurrently.
            ready = self._receive_blocking(timeout=30.0)
            if ready.get("type") != "ready":
                raise ConnectionError(
                    f"Unexpected message from server: {ready}"
                )
            self._server_capabilities = ready.get("capabilities", {})

            # --- Now start the receiver thread for ongoing messages ---
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop, daemon=True
            )
            self._receiver_thread.start()

            self._connected = True
            self.connected.emit()
            return True

        except Exception as e:
            logger.exception("Failed to connect to remote server")
            self._cleanup()
            self.error.emit(f"Connection failed: {e}")
            return False

    def disconnect_session(self):
        """Terminate remote session and close SSH connection.

        Named ``disconnect_session`` to avoid conflict with QObject.disconnect().
        """
        if not self._connected:
            return

        # Send shutdown command (non-blocking best effort)
        try:
            self._send({"request_id": 0, "command": "shutdown", "params": {}})
            time.sleep(0.5)  # Give server time to respond
        except Exception:
            pass

        self._cleanup()
        self._connected = False
        self.disconnected.emit()

    def _cleanup(self):
        """Close all connections and stop threads."""
        # Stop threads by sending None sentinel
        try:
            self._message_queue.put_nowait(None)
        except Exception:
            pass

        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5)
        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=5)

        # Clean up remote script
        if self._sftp and self._remote_script_path:
            try:
                self._sftp.remove(self._remote_script_path)
            except Exception:
                pass

        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None

        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None

        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None

        self._pending_callbacks.clear()
        self._pending_progress.clear()

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def send_command(
        self,
        command: str,
        params: dict,
        callback: Callable | None = None,
        progress_callback: Callable | None = None,
    ) -> int:
        """Send a command to the remote server.

        Parameters
        ----------
        command : str
            Command name (e.g. ``"ping"``, ``"load_deskew"``).
        params : dict
            Command-specific parameters.
        callback : callable, optional
            Called with the response dict when the command completes.
        progress_callback : callable, optional
            Called with ``(message, percentage)`` for progress updates.

        Returns
        -------
        int
            The request_id assigned to this command.
        """
        if not self._connected:
            raise RuntimeError("Not connected to remote server")

        with self._lock:
            self._request_id += 1
            request_id = self._request_id

        logger.info("[SEND_COMMAND] request_id=%s, command=%s, has_callback=%s",
                     request_id, command, callback is not None)
        if callback:
            self._pending_callbacks[request_id] = callback
        if progress_callback:
            self._pending_progress[request_id] = progress_callback

        message = {
            "request_id": request_id,
            "command": command,
            "params": params,
        }
        self._send(message)
        logger.info("[SEND_COMMAND] Message queued. Pending callbacks: %s",
                     list(self._pending_callbacks.keys()))
        return request_id

    def send_command_blocking(
        self,
        command: str,
        params: dict,
        timeout: float = 300.0,
    ) -> dict:
        """Send a command and block until the response arrives.

        Parameters
        ----------
        command : str
            Command name.
        params : dict
            Command-specific parameters.
        timeout : float
            Maximum seconds to wait for a response.

        Returns
        -------
        dict
            The response dict from the server.

        Raises
        ------
        RuntimeError
            If not connected or the command times out / fails.
        """
        if not self._connected:
            raise RuntimeError("Not connected to remote server")

        result_queue: queue.Queue[dict] = queue.Queue()

        def _cb(response: dict):
            result_queue.put(response)

        request_id = self.send_command(command, params, callback=_cb)

        try:
            response = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Command '{command}' (request_id={request_id}) timed out "
                f"after {timeout}s"
            )

        if not response.get("success", False):
            raise RuntimeError(
                f"Remote command '{command}' failed: {response.get('error')}"
            )

        return response.get("result", {})

    # ------------------------------------------------------------------
    # SFTP operations
    # ------------------------------------------------------------------

    def list_directory(self, remote_path: str) -> list[dict]:
        """List directory contents via SFTP.

        Returns
        -------
        list[dict]
            Each dict has keys: ``name``, ``is_dir``, ``size``.
        """
        if not self._sftp:
            raise RuntimeError("SFTP client not available")

        import stat as _stat

        entries: list[dict] = []
        for attr in self._sftp.listdir_attr(remote_path):
            mode = attr.st_mode if attr.st_mode is not None else 0
            entries.append({
                "name": attr.filename,
                "is_dir": _stat.S_ISDIR(mode),
                "size": attr.st_size,
            })
        return entries

    def get_sftp_client(self) -> paramiko.SFTPClient:
        """Return the active SFTP client for file browser operations."""
        if not self._sftp:
            raise RuntimeError("Not connected")
        return self._sftp

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """Return True if SSH connection is active."""
        return self._connected

    @property
    def server_capabilities(self) -> dict:
        """Return capabilities reported by the server on startup."""
        return self._server_capabilities

    # ------------------------------------------------------------------
    # Internal: sender thread
    # ------------------------------------------------------------------

    def _send(self, message: dict):
        """Enqueue a message for the sender thread."""
        self._message_queue.put(message)

    def _sender_loop(self):
        """Background thread: read from queue and write to SSH channel."""
        channel = self._channel
        if channel is None:
            logger.warning("[SENDER] Channel is None, exiting sender loop")
            return
        logger.info("[SENDER] Sender loop started")
        try:
            while True:
                message = self._message_queue.get()
                if message is None:
                    logger.info("[SENDER] Received sentinel, stopping sender loop")
                    break  # Sentinel to stop
                logger.info("[SENDER] Sending message: request_id=%s, command=%s",
                            message.get("request_id"), message.get("command"))
                payload: bytes = msgpack.packb(message, default=_encode_array)  # type: ignore[assignment]
                header = struct.pack(">Q", len(payload))
                try:
                    channel.send(header + payload)
                    logger.info("[SENDER] Message sent successfully")
                except Exception as e:
                    logger.error("[SENDER] Send error: %s", e, exc_info=True)
                    break
        except Exception as e:
            logger.error("[SENDER] Unexpected exception in sender loop: %s", e, exc_info=True)
        finally:
            logger.info("[SENDER] Sender loop exiting")

    # ------------------------------------------------------------------
    # Internal: receiver thread
    # ------------------------------------------------------------------

    def _receiver_loop(self):
        """Background thread: read from SSH channel and dispatch responses."""
        channel = self._channel
        if channel is None:
            logger.warning("[RECEIVER] Channel is None, exiting receiver loop")
            return
        logger.info("[RECEIVER] Receiver loop started")
        try:
            while True:
                message = self._receive_message_channel(channel)
                if message is None:
                    logger.warning("[RECEIVER] Received None from channel, channel closed")
                    break  # Channel closed

                logger.info("[RECEIVER] Received message: %s", message)
                msg_type = message.get("type")
                if msg_type == "ready":
                    logger.info("[RECEIVER] Ignoring duplicate ready message")
                    continue  # Already handled in connect()

                rid = message.get("request_id") or 0
                logger.info("[RECEIVER] Processing response for request_id=%s", rid)

                # Progress update
                if "progress" in message:
                    cb = self._pending_progress.get(rid)
                    if cb:
                        pct = message.get("percentage", -1)
                        if isinstance(pct, int):
                            cb(message["progress"], pct)
                        else:
                            cb(message["progress"], -1)
                    # Also emit Qt signal for UI progress bars
                    pct_val = message.get("percentage")
                    if isinstance(pct_val, int):
                        self.progress.emit(message["progress"], pct_val)
                    else:
                        self.progress.emit(message["progress"], 0)
                    continue

                # Final response
                cb = self._pending_callbacks.pop(rid, None)
                if cb:
                    logger.info("[RECEIVER] Found callback for request_id=%s, invoking", rid)
                    try:
                        cb(message)
                        logger.info("[RECEIVER] Callback completed for request_id=%s", rid)
                    except Exception as e:
                        logger.error("[RECEIVER] Callback exception for request_id=%s: %s", rid, e, exc_info=True)
                else:
                    logger.warning("[RECEIVER] No callback found for request_id=%s! Pending keys: %s",
                                   rid, list(self._pending_callbacks.keys()))
                # Also emit Qt signal for responses (thread-safe, marshaled to main thread)
                self.command_response.emit(message)
                self._pending_progress.pop(rid, None)

        except (EOFError, ConnectionError, OSError) as e:
            logger.warning("[RECEIVER] Receiver loop ended with expected exception: %s", e, exc_info=True)
        except Exception as e:
            logger.error("[RECEIVER] Receiver loop crashed with unexpected exception: %s", e, exc_info=True)
        finally:
            logger.info("[RECEIVER] Receiver loop exiting, was_connected=%s", self._connected)
            # If we were connected, emit disconnected
            if self._connected:
                self._connected = False
                self.disconnected.emit()

    def _receive_message_channel(self, channel: paramiko.Channel) -> dict | None:
        """Read a single length-prefixed MessagePack message from a channel."""
        logger.debug("[RECEIVE_MSG] Reading header (8 bytes)...")
        header = self._read_exact_channel(channel, 8)
        if len(header) < 8:
            logger.warning("[RECEIVE_MSG] Header too short (%d bytes), channel closed", len(header))
            return None  # Channel closed
        length = struct.unpack(">Q", header)[0]
        logger.debug("[RECEIVE_MSG] Payload length=%d, reading data...", length)
        data = self._read_exact_channel(channel, length)
        if len(data) < length:
            logger.warning("[RECEIVE_MSG] Data too short (%d < %d bytes)", len(data), length)
            return None
        try:
            msg = msgpack.unpackb(data, strict_map_key=False, object_hook=_decode_array)
            logger.debug("[RECEIVE_MSG] Successfully unpacked message")
            return msg
        except Exception as e:
            logger.error("[RECEIVE_MSG] Failed to unpack MessagePack: %s (data length=%d)", e, len(data), exc_info=True)
            raise

    def _receive_blocking(self, timeout: float = 30.0) -> dict:
        """Block until a message is received (used for ready message)."""
        result: queue.Queue[dict] = queue.Queue()
        channel = self._channel

        def _reader():
            if channel is None:
                return
            try:
                msg = self._receive_message_channel(channel)
                if msg:
                    result.put(msg)
            except Exception as e:
                result.put({"type": "error", "error": str(e)})

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            return result.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("Timed out waiting for server ready message")

    @staticmethod
    def _read_exact_channel(channel: paramiko.Channel, n: int) -> bytes:
        """Read exactly *n* bytes from a paramiko Channel."""
        data = b""
        while len(data) < n:
            try:
                chunk = channel.recv(min(65536, n - len(data)))
            except Exception as e:
                logger.warning("[READ_EXACT] Exception reading %d bytes: %s", n - len(data), e)
                break
            if not chunk:
                logger.warning("[READ_EXACT] Empty chunk received, total so far: %d / %d", len(data), n)
                break
            data += chunk
        return data
