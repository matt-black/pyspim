"""
SFTP directory browser dialog for napari-pyspim.

Provides a simple file browser for navigating remote directories
via SFTP when in remote mode.
"""

import os
from pathlib import Path

from qtpy.QtCore import Qt, QSortFilterProxyModel
from qtpy.QtGui import QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class SftpBrowserDialog(QDialog):
    """Dialog for browsing remote directories via SFTP.

    Args:
        remote_client: The RemoteClient instance with active SSH connection.
        initial_path: Starting directory path on the remote server.
        parent: Parent widget.
    """

    def __init__(self, remote_client, initial_path=None, parent=None):
        super().__init__(parent)
        self._remote_client = remote_client
        self._current_path = initial_path or '/'
        self.selected_path = None
        self.setWindowTitle("Select Remote Directory")
        self.setMinimumSize(600, 400)
        self.setup_ui()
        self._load_directory(self._current_path)

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Current path display
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Current Path:"))
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setStyleSheet("background-color: #f0f0f0;")
        path_layout.addWidget(self.path_edit)

        go_button = QPushButton("Go")
        go_button.clicked.connect(self._go_to_path)
        path_layout.addWidget(go_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(lambda: self._load_directory(self._current_path))
        path_layout.addWidget(refresh_button)

        layout.addLayout(path_layout)

        # Directory listing
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget)

        # Buttons
        button_layout = QHBoxLayout()

        self.up_button = QPushButton("..")
        self.up_button.clicked.connect(self._go_up)
        self.up_button.setToolTip("Go to parent directory")
        button_layout.addWidget(self.up_button)

        button_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.select_button = QPushButton("Select Directory")
        self.select_button.clicked.connect(self._select_directory)
        self.select_button.setDefault(True)
        button_layout.addWidget(self.select_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _load_directory(self, path):
        """Load and display the contents of a remote directory."""
        try:
            self._current_path = path
            self.path_edit.setText(path)

            # Use SFTP to list directory
            sftp = self._remote_client.sftp_client
            if sftp is None:
                QMessageBox.warning(self, "Error", "Not connected to remote server.")
                return

            entries = []
            try:
                for attr in sftp.listdir_attr(path):
                    name = attr.filename
                    is_dir = attr.st_mode is not None and (
                        attr.st_mode & 0o170000 == 0o040000
                    )
                    size = attr.st_size if not is_dir else 0
                    entries.append((name, is_dir, size))
            except IOError as e:
                QMessageBox.warning(self, "Error", f"Cannot list directory: {e}")
                return

            # Sort: directories first, then files, alphabetically within each group
            dirs = sorted([e for e in entries if e[1]], key=lambda x: x[0].lower())
            files = sorted([e for e in entries if not e[1]], key=lambda x: x[0].lower())
            sorted_entries = dirs + files

            # Populate list widget
            self.list_widget.clear()
            for name, is_dir, size in sorted_entries:
                item = QListWidgetItem(self.list_widget)
                if is_dir:
                    item.setText(f"📁  {name}/")
                    item.setToolTip(f"Directory: {os.path.join(path, name)}")
                else:
                    item.setText(f"📄  {name}")
                    item.setToolTip(f"File: {self._format_size(size)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load directory: {e}")

    def _go_to_path(self):
        """Navigate to the path in the edit field."""
        path = self.path_edit.text().strip()
        if path:
            self._load_directory(path)

    def _go_up(self):
        """Navigate to the parent directory."""
        parent = os.path.dirname(self._current_path)
        if parent != self._current_path:
            self._load_directory(parent)
        else:
            self._load_directory('/')

    def _on_item_double_clicked(self, item):
        """Handle double-click on a list item."""
        text = item.text()
        # Remove icon prefix
        name = text.split('  ', 1)[-1].rstrip('/')

        full_path = os.path.join(self._current_path, name)

        # Check if it's a directory
        sftp = self._remote_client.sftp_client
        try:
            attr = sftp.stat(full_path)
            is_dir = attr.st_mode & 0o170000 == 0o040000
            if is_dir:
                self._load_directory(full_path)
        except IOError:
            pass

    def _select_directory(self):
        """Handle directory selection."""
        # Get the currently selected item
        current_item = self.list_widget.currentItem()
        if current_item is None:
            # If nothing selected, select current path
            self.selected_path = self._current_path
            self.accept()
            return

        text = current_item.text()
        name = text.split('  ', 1)[-1].rstrip('/')
        full_path = os.path.join(self._current_path, name)

        # Verify it's a directory
        sftp = self._remote_client.sftp_client
        try:
            attr = sftp.stat(full_path)
            is_dir = attr.st_mode & 0o170000 == 0o040000
            if is_dir:
                self.selected_path = full_path
                self.accept()
            else:
                QMessageBox.warning(self, "Not a Directory",
                                    f"'{name}' is not a directory.")
        except IOError as e:
            QMessageBox.warning(self, "Error", f"Cannot access '{name}': {e}")

    @staticmethod
    def _format_size(size_bytes):
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    @staticmethod
    def browse(remote_client, initial_path=None, parent=None):
        """Convenience method to show the browser dialog.

        Args:
            remote_client: The RemoteClient instance.
            initial_path: Starting directory path.
            parent: Parent widget.

        Returns:
            Selected directory path, or None if cancelled.
        """
        dialog = SftpBrowserDialog(remote_client, initial_path, parent)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            return dialog.selected_path
        return None
