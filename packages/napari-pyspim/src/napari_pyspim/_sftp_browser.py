"""SFTP browser dialog for navigating remote directories.

Provides a simple file browser that uses an active SFTP connection
to list and navigate remote directories.
"""

from __future__ import annotations

import os
import stat as _stat
from typing import Optional

import paramiko
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
)


class _DirectoryModel(QStandardItemModel):  # type: ignore[name-defined]
    """Simple model for displaying directory listings."""

    def __init__(self, parent=None):  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self._current_path: str = "/"
        self.setHorizontalHeaderLabels(["Name", "Type", "Size"])

    @property
    def current_path(self) -> str:
        return self._current_path

    def set_data(self, path: str, entries: list[dict]):
        """Update model with new directory listing."""
        self._current_path = path
        self.removeRows(0, self.rowCount())
        for entry in entries:
            name_item = QStandardItem(entry["name"])
            type_item = QStandardItem("Dir" if entry["is_dir"] else "File")
            size = entry.get("size", 0) or 0
            if entry["is_dir"]:
                size_item = QStandardItem("")
            else:
                size_item = QStandardItem(self._format_size(size))
            self.appendRow([name_item, type_item, size_item])

    @staticmethod
    def _format_size(size: int) -> str:
        s: float = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if abs(s) < 1024:
                return f"{s:.1f} {unit}"
            s /= 1024
        return f"{s:.1f} PB"


class SftpBrowserDialog(QDialog):
    """Dialog for browsing remote directories via SFTP.

    Parameters
    ----------
    sftp_client : paramiko.SFTPClient
        Active SFTP client.
    parent : QWidget, optional
        Parent widget.
    initial_path : str
        Starting directory path.
    title : str
        Dialog title.
    """

    def __init__(
        self,
        sftp_client: paramiko.SFTPClient,
        parent=None,  # type: ignore[no-untyped-def]
        initial_path: str = "/",
        title: str = "Browse Remote Directory",
    ):
        super().__init__(parent)
        self.sftp = sftp_client
        self.selected_path: Optional[str] = None
        self._current_path = initial_path
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)
        self._model = _DirectoryModel(self)
        self._setup_ui()
        self._navigate_to(initial_path)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Path bar
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Path:"))
        self.path_edit = QLineEdit()
        self.path_edit.returnPressed.connect(self._on_path_return)
        path_layout.addWidget(self.path_edit)
        self.up_button = QPushButton("↑ Up")
        self.up_button.clicked.connect(self._go_up)
        path_layout.addWidget(self.up_button)
        layout.addLayout(path_layout)

        # Directory listing
        self.table = QTableView()
        self.table.setModel(self._model)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.doubleClicked.connect(self._on_double_click)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.Fixed)
            header.setSectionResizeMode(2, QHeaderView.Fixed)
            header.resizeSection(1, 60)
            header.resizeSection(2, 100)
        layout.addWidget(self.table)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(self._on_select)
        button_layout.addWidget(self.select_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

    def _navigate_to(self, path: str):
        """Navigate to *path* and refresh the listing."""
        try:
            # Normalize path
            path = path.rstrip("/") or "/"
            self._current_path = path
            self.path_edit.setText(path)

            # List directory
            entries: list[dict] = []
            for attr in self.sftp.listdir_attr(path):
                mode = attr.st_mode if attr.st_mode is not None else 0
                entries.append({
                    "name": attr.filename,
                    "is_dir": _stat.S_ISDIR(mode),
                    "size": attr.st_size,
                })

            self._model.set_data(path, entries)
            self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # type: ignore[attr-defined]
        except Exception as e:
            self.path_edit.setText(f"Error: {e}")

    def _go_up(self):
        parts = self._current_path.split("/")
        if len(parts) > 1 and parts[-1]:
            parts.pop()
        elif self._current_path != "/":
            parts = ["/"]
        path = "/".join(parts) or "/"
        self._navigate_to(path)

    def _on_path_return(self):
        path = self.path_edit.text().strip()
        if path:
            self._navigate_to(path)

    def _on_double_click(self, index):
        name = self._model.data(index, Qt.ItemDataRole.DisplayRole)  # type: ignore[attr-defined]
        is_dir = self._model.data(index.sibling(index.row(), 1), Qt.ItemDataRole.DisplayRole) == "Dir"  # type: ignore[attr-defined]
        if is_dir:
            new_path = os.path.join(self._current_path, name)
            self._navigate_to(new_path)

    def _on_select(self):
        """Accept the current directory as the selected path."""
        self.selected_path = self._current_path
        self.accept()

    def keyPressEvent(self, event):  # type: ignore[override]
        if event.key() in (Qt.Key_Backspace,):  # type: ignore[attr-defined]
            self._go_up()
        super().keyPressEvent(event)
