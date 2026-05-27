"""Remote Connection tab for configuring and managing SSH connections.

Provides the UI for Tab 0 in the main pipeline widget, allowing users to
configure SSH connection parameters and connect/disconnect from a remote
compute server.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._remote_client import RemoteClient
from ._sftp_browser import SftpBrowserDialog


class RemoteConnectionWidget(QWidget):
    """Tab 0 widget for SSH connection management.

    Parameters
    ----------
    napari_viewer
        The napari viewer instance (passed by plugin infrastructure).
    remote_client : RemoteClient, optional
        Pre-existing client to reuse.  If ``None`` a new one is created.
    """

    def __init__(
        self,
        napari_viewer,  # type: ignore[name-defined]
        remote_client: Optional[RemoteClient] = None,
    ):
        super().__init__()
        self.viewer = napari_viewer
        self.client = remote_client or RemoteClient()
        self._connecting = False
        self._waiting_for_test = False  # Track if we're waiting for test connection response
        self._setup_ui()
        self._load_config()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Connection Settings Group ---
        settings_group = QGroupBox("SSH Connection Settings")
        settings_layout = QFormLayout()

        # Host
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("e.g. compute.example.com or 192.168.1.100")
        settings_layout.addRow("Host:", self.host_edit)

        # Port + Username (same row)
        port_user_layout = QHBoxLayout()
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        port_user_layout.addWidget(QLabel("Port:"))
        port_user_layout.addWidget(self.port_spin)
        port_user_layout.addSpacing(20)
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("SSH username")
        port_user_layout.addWidget(QLabel("Username:"))
        port_user_layout.addWidget(self.username_edit)
        settings_layout.addRow("", port_user_layout)

        # Auth method
        auth_layout = QHBoxLayout()
        self.auth_password = QRadioButton("Password")
        self.auth_key = QRadioButton("Private Key")
        self.auth_key.setChecked(True)
        auth_layout.addWidget(self.auth_password)
        auth_layout.addWidget(self.auth_key)
        settings_layout.addRow("Auth Method:", auth_layout)

        # Password
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.Password)
        self.password_edit.setVisible(False)
        settings_layout.addRow("Password:", self.password_edit)

        # Private key path + browse
        key_layout = QHBoxLayout()
        self.key_path_edit = QLineEdit()
        self.key_path_edit.setPlaceholderText("/path/to/local/private/key")
        self.key_browse_btn = QPushButton("Browse")
        key_layout.addWidget(self.key_path_edit)
        key_layout.addWidget(self.key_browse_btn)
        settings_layout.addRow("Private Key:", key_layout)

        # Key passphrase
        self.key_passphrase_edit = QLineEdit()
        self.key_passphrase_edit.setEchoMode(QLineEdit.Password)
        self.key_passphrase_edit.setPlaceholderText("Optional")
        settings_layout.addRow("Key Passphrase:", self.key_passphrase_edit)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # --- Remote Environment Group ---
        env_group = QGroupBox("Remote Environment")
        env_layout = QFormLayout()

        venv_layout = QHBoxLayout()
        self.venv_path_edit = QLineEdit()
        self.venv_path_edit.setPlaceholderText(
            "e.g. /home/user/venv  (path to pyspim venv on remote server)"
        )
        self.venv_browse_btn = QPushButton("Browse")
        venv_layout.addWidget(self.venv_path_edit)
        venv_layout.addWidget(self.venv_browse_btn)
        env_layout.addRow("Remote Python Venv:", venv_layout)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # --- Status + Progress ---
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # --- Connect / Disconnect + Test Buttons ---
        button_layout = QHBoxLayout()

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setMinimumHeight(40)
        self.connect_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        button_layout.addWidget(self.connect_btn)

        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setMinimumHeight(40)
        self.test_btn.setEnabled(False)
        self.test_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        button_layout.addWidget(self.test_btn)

        layout.addLayout(button_layout)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signal Connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Auth method toggle
        self.auth_password.toggled.connect(self._on_auth_method_changed)

        # Browse buttons
        self.key_browse_btn.clicked.connect(self._on_key_browse)
        self.venv_browse_btn.clicked.connect(self._on_venv_browse)

        # Connect button
        self.connect_btn.clicked.connect(self._on_connect_clicked)

        # Test button
        self.test_btn.clicked.connect(self._on_test_clicked)

        # Remote client signals
        self.client.connected.connect(self._on_connected)
        self.client.disconnected.connect(self._on_disconnected)
        self.client.error.connect(self._on_error)
        self.client.progress.connect(self._on_progress)
        self.client.command_response.connect(self._on_command_response)

    # ------------------------------------------------------------------
    # Config Persistence
    # ------------------------------------------------------------------

    def _config_path(self) -> Path:
        return Path.home() / ".pyspim" / "remote_config.json"

    def _load_config(self):
        """Load saved connection settings (not password/passphrase)."""
        cfg = self._config_path()
        if not cfg.exists():
            return
        try:
            data = json.loads(cfg.read_text())
            self.host_edit.setText(data.get("host", ""))
            self.port_spin.setValue(data.get("port", 22))
            self.username_edit.setText(data.get("username", ""))
            method = data.get("auth_method", "key")
            if method == "password":
                self.auth_password.setChecked(True)
            else:
                self.auth_key.setChecked(True)
            self.key_path_edit.setText(data.get("key_path", ""))
            self.venv_path_edit.setText(data.get("remote_venv", ""))
        except (json.JSONDecodeError, OSError):
            pass

    def _save_config(self):
        """Persist non-sensitive connection settings."""
        cfg = self._config_path()
        cfg.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "host": self.host_edit.text().strip(),
            "port": self.port_spin.value(),
            "username": self.username_edit.text().strip(),
            "auth_method": "password" if self.auth_password.isChecked() else "key",
            "key_path": self.key_path_edit.text().strip(),
            "remote_venv": self.venv_path_edit.text().strip(),
        }
        cfg.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Slot Implementations
    # ------------------------------------------------------------------

    def _on_auth_method_changed(self):
        show_password = self.auth_password.isChecked()
        self.password_edit.setVisible(show_password)
        self.key_path_edit.setVisible(not show_password)
        self.key_browse_btn.setVisible(not show_password)
        self.key_passphrase_edit.setVisible(not show_password)

    def _on_key_browse(self):
        """Open local file dialog for private key selection."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SSH Private Key",
            str(Path.home() / ".ssh"),
            "Private Keys (*.pem *.key *id_rsa *id_ed25519 *);;All Files (*)",
        )
        if path:
            self.key_path_edit.setText(path)

    def _on_venv_browse(self):
        """Open SFTP browser for remote venv path selection."""
        if not self.client.is_connected:
            QMessageBox.information(
                self,
                "Not Connected",
                "You must connect to the server first before browsing "
                "remote directories.\n\nAlternatively, type the path "
                "manually and then click Connect.",
            )
            return
        try:
            sftp = self.client.get_sftp_client()
            # Determine home directory
            try:
                home = sftp.normalize(".")
            except Exception:
                home = "/"
            dialog = SftpBrowserDialog(
                sftp,
                parent=self,
                initial_path=home,
                title="Select Remote Python Virtual Environment",
            )
            if dialog.exec_() == QDialog.Accepted and dialog.selected_path:  # type: ignore[attr-defined]
                self.venv_path_edit.setText(dialog.selected_path)
        except Exception as e:
            QMessageBox.critical(self, "SFTP Error", str(e))

    def _on_connect_clicked(self):
        if self.client.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """Validate inputs and attempt SSH connection."""
        host = self.host_edit.text().strip()
        port = self.port_spin.value()
        username = self.username_edit.text().strip()

        if not host:
            QMessageBox.warning(self, "Missing Host", "Please enter a host.")
            return
        if not username:
            QMessageBox.warning(self, "Missing Username", "Please enter a username.")
            return

        auth_method = "password" if self.auth_password.isChecked() else "key"
        password = self.password_edit.text() if auth_method == "password" else None
        key_path = self.key_path_edit.text().strip() if auth_method == "key" else None
        key_passphrase = self.key_passphrase_edit.text() if auth_method == "key" else None
        remote_venv = self.venv_path_edit.text().strip() or None

        # Save config before connecting
        self._save_config()

        # Update UI to connecting state
        self._set_connecting_state()

        # Use QTimer to defer connection so UI can update
        QTimer.singleShot(50, lambda: self._do_connect(
            host, port, username, auth_method,
            password, key_path, key_passphrase, remote_venv,
        ))

    def _do_connect(
        self, host, port, username, auth_method,
        password, key_path, key_passphrase, remote_venv,
    ):
        success = self.client.connect(
            host=host,
            port=port,
            username=username,
            auth_method=auth_method,
            password=password,
            key_path=key_path,
            key_passphrase=key_passphrase,
            remote_venv=remote_venv,
        )
        if not success:
            # Reset to disconnected state (error signal may also fire)
            self._set_disconnected_state()

    def _disconnect(self):
        self._set_disconnecting_state()
        self.client.disconnect_session()

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def _set_connecting_state(self):
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Connecting...")
        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self._set_inputs_enabled(False)

    def _set_connected_state(self):
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Disconnect")
        self.connect_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; "
            "background-color: #e74c3c; color: white; }"
        )
        self.status_label.setText("Connected")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        self.progress_bar.setVisible(False)
        self._set_inputs_enabled(True)
        self.test_btn.setEnabled(True)

        # Show capabilities
        caps = self.client.server_capabilities
        if caps:
            cuda = caps.get("has_cuda", "unknown")
            version = caps.get("pyspim_version", "unknown")
            self.status_label.setText(
                f"Connected  |  CUDA: {cuda}  |  pyspim: {version}"
            )

    def _set_disconnected_state(self):
        self.connect_btn.setEnabled(True)
        self.connect_btn.setText("Connect")
        self.connect_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
        )
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: gray; font-weight: bold;")
        self.progress_bar.setVisible(False)
        self._set_inputs_enabled(True)
        self.test_btn.setEnabled(False)

    def _set_disconnecting_state(self):
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("Disconnecting...")
        self.status_label.setText("Disconnecting...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

    def _set_inputs_enabled(self, enabled: bool):
        self.host_edit.setEnabled(enabled)
        self.port_spin.setEnabled(enabled)
        self.username_edit.setEnabled(enabled)
        self.auth_password.setEnabled(enabled)
        self.auth_key.setEnabled(enabled)
        self.password_edit.setEnabled(enabled)
        self.key_path_edit.setEnabled(enabled)
        self.key_browse_btn.setEnabled(enabled)
        self.key_passphrase_edit.setEnabled(enabled)
        self.venv_path_edit.setEnabled(enabled)
        # Venv browse only works when connected
        self.venv_browse_btn.setEnabled(self.client.is_connected)

    # ------------------------------------------------------------------
    # Client Signal Handlers
    # ------------------------------------------------------------------

    def _on_connected(self):
        self._set_connected_state()

    def _on_disconnected(self):
        self._set_disconnected_state()

    def _on_error(self, message: str):
        self._set_disconnected_state()
        self.status_label.setText("Connection Failed")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        QMessageBox.critical(self, "Connection Error", message)

    def _on_progress(self, message: str, percentage: int):
        if percentage >= 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)
        self.status_label.setText(message)

    # ------------------------------------------------------------------
    # Test Connection
    # ------------------------------------------------------------------

    def _on_test_clicked(self):
        """Send a test command to the remote server and display the result."""
        if not self.client.is_connected:
            QMessageBox.warning(self, "Not Connected", "Please connect first.")
            return

        self.test_btn.setEnabled(False)
        self.test_btn.setText("Testing...")
        self.status_label.setText("Testing connection...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        # Mark that we're waiting for a test response
        self._waiting_for_test = True

        # Safety timeout: reset button if no response within 15 seconds
        self._test_timeout_timer = QTimer(self)
        self._test_timeout_timer.setSingleShot(True)
        self._test_timeout_timer.timeout.connect(self._on_test_timeout)
        self._test_timeout_timer.start(15000)

        self.client.send_command(
            command="compute",
            params={"expression": "2+2"},
            callback=None,  # Use command_response signal instead
            progress_callback=None,
        )

    def _on_test_timeout(self):
        """Called when the test connection times out."""
        self._waiting_for_test = False
        self.test_btn.setEnabled(True)
        self.test_btn.setText("Test Connection")
        self.status_label.setText("Test timed out")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        QMessageBox.critical(
            self,
            "Test Timed Out",
            "No response from the remote server within 15 seconds.\n\n"
            "The server may have crashed. Try reconnecting.",
        )

    def _on_command_response(self, response: dict):
        """Handle command responses from the server (runs on main thread via Qt signal).

        This slot is connected to RemoteClient.command_response which is emitted
        from the receiver thread. Qt automatically marshals the signal to the
        main thread when the receiver lives on the main thread.

        Only handles 'compute' command responses (from Test Connection button).
        Other commands (compute_projections, query_positions) are handled by
        their respective widgets via _pending_command_type.
        """
        # Only handle responses when we're waiting for a test connection result
        if not self._waiting_for_test:
            return
        self._waiting_for_test = False

        # Cancel the timeout timer if it's still pending
        timer = getattr(self, "_test_timeout_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
            del self._test_timeout_timer

        if response.get("success"):
            result = response.get("result", {})
            expression = result.get("expression", "2+2")
            value = result.get("result", "unknown")
            QMessageBox.information(
                self,
                "Connection Test Successful",
                f"Remote calculation completed successfully!\n\n"
                f"{expression} = {value}",
            )
            self.status_label.setText("Test passed")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            error_msg = response.get("error", "Unknown error")
            QMessageBox.critical(
                self,
                "Connection Test Failed",
                f"The remote server returned an error:\n\n{error_msg}",
            )
            self.status_label.setText("Test failed")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        self.test_btn.setEnabled(True)
        self.test_btn.setText("Test Connection")
