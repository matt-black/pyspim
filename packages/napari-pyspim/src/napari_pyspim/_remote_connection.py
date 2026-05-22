"""
Remote connection widget for napari-pyspim.

Provides a tab for configuring and managing SSH connections to a
remote compute server.
"""

from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
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


class RemoteConnectionWidget(QWidget):
    """Widget for configuring and managing remote SSH connection."""

    def __init__(self, remote_client: RemoteClient, parent=None):
        super().__init__(parent)
        self._client = remote_client
        self._client.connected.connect(self._on_connected)
        self._client.disconnected.connect(self._on_disconnected)
        self._client.error.connect(self._on_error)
        self._client.progress.connect(self._on_progress)
        self._connecting = False
        self.setup_ui()
        self._load_saved_config()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Connection Settings Group
        settings_group = QGroupBox("Connection Settings")
        settings_layout = QFormLayout()

        # Host
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("e.g., compute.example.com")
        settings_layout.addRow("Host:", self.host_edit)

        # Port
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(22)
        settings_layout.addRow("Port:", self.port_spin)

        # Username
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("SSH username")
        settings_layout.addRow("Username:", self.username_edit)

        # Auth Method
        auth_layout = QHBoxLayout()
        self.auth_password = QRadioButton("Password")
        self.auth_key = QRadioButton("Private Key")
        self.auth_key.setChecked(True)
        self.auth_password.toggled.connect(self._on_auth_method_changed)
        auth_layout.addWidget(self.auth_password)
        auth_layout.addWidget(self.auth_key)
        auth_layout.addStretch()
        settings_layout.addRow("Auth Method:", auth_layout)

        # Password
        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("SSH password")
        self.password_edit.setVisible(False)
        settings_layout.addRow("Password:", self.password_edit)

        # Private Key Path
        key_layout = QHBoxLayout()
        self.key_path_edit = QLineEdit()
        self.key_path_edit.setPlaceholderText("Path to SSH private key")
        self.key_browse_button = QPushButton("Browse")
        self.key_browse_button.clicked.connect(self._browse_key_path)
        key_layout.addWidget(self.key_path_edit)
        key_layout.addWidget(self.key_browse_button)
        settings_layout.addRow("Private Key:", key_layout)

        # Key Passphrase
        self.key_passphrase_edit = QLineEdit()
        self.key_passphrase_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_passphrase_edit.setPlaceholderText("Key passphrase (optional)")
        settings_layout.addRow("Key Passphrase:", self.key_passphrase_edit)

        settings_group.setLayout(settings_layout)

        # Remote Environment Group
        env_group = QGroupBox("Remote Environment")
        env_layout = QFormLayout()

        self.venv_edit = QLineEdit()
        self.venv_edit.setPlaceholderText("e.g., /home/user/venv")
        self.venv_edit.setToolTip(
            "Path to the Python virtual environment on the remote server "
            "where pyspim is installed."
        )
        env_layout.addRow("Python Virtual Env:", self.venv_edit)

        env_group.setLayout(env_layout)

        # Connection Status
        status_layout = QHBoxLayout()

        self.status_label = QLabel("Disconnected")
        self.status_label.setStyleSheet("color: #888; font-weight: bold;")
        status_layout.addWidget(self.status_label)

        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(12, 12)
        self.status_indicator.setStyleSheet(
            "background-color: #cc0000; border-radius: 6px;"
        )
        self.status_indicator.setToolTip("Connection status")
        status_layout.addWidget(self.status_indicator)

        status_layout.addStretch()

        # Connect/Disconnect Button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._toggle_connection)
        self.connect_button.setMinimumWidth(100)
        status_layout.addWidget(self.connect_button)

        # Save Config Button
        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.clicked.connect(self._save_config)
        self.save_config_button.setToolTip("Save connection settings for next session")
        status_layout.addWidget(self.save_config_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        # Add to main layout
        layout.addWidget(settings_group)
        layout.addWidget(env_group)
        layout.addLayout(status_layout)
        layout.addWidget(self.progress_bar)
        layout.addStretch()

        self.setLayout(layout)

    def _on_auth_method_changed(self):
        """Toggle visibility of password/key fields based on auth method."""
        use_password = self.auth_password.isChecked()
        self.password_edit.setVisible(use_password)
        self.key_path_edit.setVisible(not use_password)
        self.key_browse_button.setVisible(not use_password)
        self.key_passphrase_edit.setVisible(not use_password)

    def _browse_key_path(self):
        """Open file dialog to select SSH private key."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key", str(Path.home() / '.ssh'),
            "Key Files (*.pem *.key *id_rsa *id_ed25519 *);;All Files (*)"
        )
        if path:
            self.key_path_edit.setText(path)

    def _load_saved_config(self):
        """Load saved connection settings into the UI."""
        config = self._client.load_config()
        if not config:
            return

        if 'host' in config:
            self.host_edit.setText(config['host'])
        if 'port' in config:
            self.port_spin.setValue(config['port'])
        if 'username' in config:
            self.username_edit.setText(config['username'])
        if 'auth_method' in config:
            if config['auth_method'] == 'password':
                self.auth_password.setChecked(True)
            else:
                self.auth_key.setChecked(True)
            self._on_auth_method_changed()
        if 'key_path' in config:
            self.key_path_edit.setText(config['key_path'])
        if 'remote_venv' in config:
            self.venv_edit.setText(config['remote_venv'])

    def _save_config(self):
        """Save current settings to config file."""
        config = {
            'host': self.host_edit.text(),
            'port': self.port_spin.value(),
            'username': self.username_edit.text(),
            'auth_method': 'password' if self.auth_password.isChecked() else 'key',
            'key_path': self.key_path_edit.text(),
            'remote_venv': self.venv_edit.text(),
        }
        self._client.save_config(config)
        QMessageBox.information(self, "Config Saved",
                                "Connection settings saved to ~/.pyspim/remote_config.json")

    def _toggle_connection(self):
        """Handle Connect/Disconnect button click."""
        if self._client.is_connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        """Establish SSH connection."""
        host = self.host_edit.text().strip()
        port = self.port_spin.value()
        username = self.username_edit.text().strip()

        if not host:
            QMessageBox.warning(self, "Missing Host", "Please enter a remote host.")
            return
        if not username:
            QMessageBox.warning(self, "Missing Username", "Please enter an SSH username.")
            return

        auth_method = 'password' if self.auth_password.isChecked() else 'key'
        password = self.password_edit.text() if auth_method == 'password' else None
        key_path = self.key_path_edit.text().strip() if auth_method == 'key' else None
        key_passphrase = self.key_passphrase_edit.text() if auth_method == 'key' else None
        remote_venv = self.venv_edit.text().strip() or None

        if auth_method == 'key' and not key_path:
            QMessageBox.warning(self, "Missing Key Path",
                                "Please select an SSH private key file.")
            return

        # Auto-save config on connect
        config = {
            'host': host,
            'port': port,
            'username': username,
            'auth_method': auth_method,
            'key_path': key_path or '',
            'remote_venv': remote_venv or '',
        }
        self._client.save_config(config)

        # Update UI for connecting state
        self._set_connecting_state(True)

        # Attempt connection
        self._client.connect(
            host=host,
            port=port,
            username=username,
            auth_method=auth_method,
            password=password,
            key_path=key_path,
            key_passphrase=key_passphrase,
            remote_venv=remote_venv,
        )

    def _disconnect(self):
        """Disconnect from remote server."""
        self._client.disconnect_ssh()

    def _set_connecting_state(self, connecting: bool):
        """Update UI for connecting/disconnecting state."""
        self._connecting = connecting
        self.connect_button.setEnabled(not connecting)
        self.host_edit.setEnabled(not connecting)
        self.port_spin.setEnabled(not connecting)
        self.username_edit.setEnabled(not connecting)
        self.password_edit.setEnabled(not connecting)
        self.key_path_edit.setEnabled(not connecting)
        self.key_passphrase_edit.setEnabled(not connecting)
        self.venv_edit.setEnabled(not connecting)
        self.auth_password.setEnabled(not connecting)
        self.auth_key.setEnabled(not connecting)
        self.progress_bar.setVisible(connecting)

        if connecting:
            self.connect_button.setText("Connecting...")
            self.status_label.setText("Connecting...")
            self.status_label.setStyleSheet("color: #cc9900; font-weight: bold;")
            self.status_indicator.setStyleSheet(
                "background-color: #cc9900; border-radius: 6px;"
            )
        else:
            self.connect_button.setText("Connect")

    @staticmethod
    def _on_connected():
        """Handle successful connection."""
        # UI updates handled by _on_connection_state_changed via timer
        pass

    def _on_disconnected(self):
        """Handle disconnection."""
        QTimer.singleShot(0, self._update_disconnected_ui)

    def _update_disconnected_ui(self):
        """Update UI for disconnected state on main thread."""
        self._set_connecting_state(False)
        self.connect_button.setText("Connect")
        self.status_label.setText("Disconnected")
        self.status_label.setStyleSheet("color: #888; font-weight: bold;")
        self.status_indicator.setStyleSheet(
            "background-color: #cc0000; border-radius: 6px;"
        )
        self.progress_bar.setVisible(False)

    def _on_connected_ui(self):
        """Update UI for connected state on main thread."""
        self._set_connecting_state(False)
        self.connect_button.setText("Disconnect")
        self.status_label.setText(f"Connected to {self.host_edit.text()}")
        self.status_label.setStyleSheet("color: #00aa00; font-weight: bold;")
        self.status_indicator.setStyleSheet(
            "background-color: #00cc00; border-radius: 6px;"
        )
        self.progress_bar.setVisible(False)

    def _on_error(self, message: str):
        """Handle error from remote client."""
        self._set_connecting_state(False)
        self.status_label.setText("Connection Failed")
        self.status_label.setStyleSheet("color: #cc0000; font-weight: bold;")
        self.status_indicator.setStyleSheet(
            "background-color: #cc0000; border-radius: 6px;"
        )
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Connection Error", message)

    def _on_progress(self, message: str):
        """Handle progress update."""
        self.status_label.setText(message)

    # Override signal handlers to update UI on main thread
    def _on_connected(self):
        QTimer.singleShot(0, self._on_connected_ui)
