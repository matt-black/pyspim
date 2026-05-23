"""
ROI detection widget for manual region of interest detection and cropping.
"""

import json
import os

import numpy as np
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._remote_client import RemoteClient
from ._sftp_browser import SftpBrowserDialog


class ProjectionLoaderWorker(QThread):
    """Worker thread for loading data and computing projections for both views."""

    projections_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(
        self,
        data_path: str,
        channel: int,
        projection_type: str,
        multi_pos: bool = False,
        time: int = 0,
        position: int = 0,
    ):
        super().__init__()
        self.data_path = data_path
        self.channel = channel  # 0-indexed
        self.projection_type = projection_type  # 'max', 'sum', 'mean'
        self.multi_pos = multi_pos
        self.time = time
        self.position = position

    def _compute_projections(self, volume, proj_type):
        """Compute YX and ZY projections from a volume."""
        if proj_type == "max":
            yx_proj = np.max(volume, axis=0)  # shape: (Y, X)
            zy_proj = np.max(volume, axis=2)  # shape: (Z, Y)
        elif proj_type == "sum":
            yx_proj = np.sum(volume, axis=0)
            zy_proj = np.sum(volume, axis=2)
        elif proj_type == "mean":
            yx_proj = np.mean(volume, axis=0)
            zy_proj = np.mean(volume, axis=2)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        return yx_proj, zy_proj

    def run(self):
        """Load data and compute projections for both views in background thread."""
        try:
            from pyspim.data import dispim as data

            with data.uManagerAcquisition(self.data_path, self.multi_pos, np) as acq:
                # Load volumes for both heads
                # Raw data structure: ZYX (Z is leading dimension)
                if self.multi_pos:
                    volume_a = acq.get(self.position, 'a', self.channel, self.time)
                    volume_b = acq.get(self.position, 'b', self.channel, self.time)
                else:
                    volume_a = acq.get('a', self.channel, self.time)
                    volume_b = acq.get('b', self.channel, self.time)

            # Compute projections for View A
            yx_proj_a, zy_proj_a = self._compute_projections(volume_a, self.projection_type)
            # Compute projections for View B
            yx_proj_b, zy_proj_b = self._compute_projections(volume_b, self.projection_type)

            result = {
                "yx_proj_a": yx_proj_a,
                "zy_proj_a": zy_proj_a,
                "yx_proj_b": yx_proj_b,
                "zy_proj_b": zy_proj_b,
                "volume_shape_a": volume_a.shape,
                "volume_shape_b": volume_b.shape,
                "data_path": self.data_path,
            }

            self.projections_loaded.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class RoiDetectionWidget(QWidget):
    """Widget for manual ROI detection and cropping."""

    roi_applied = Signal(dict)

    def __init__(self, viewer, remote_client: RemoteClient | None = None):
        super().__init__()
        self.viewer = viewer
        self.remote_client = remote_client
        self.loader_worker = None
        self.data_path = None
        self._remote_mode = False  # True when data_path is on remote server
        # Track pending remote command type for signal-based response handling
        self._pending_command_type = None  # 'compute_projections' or 'query_positions'
        self._pending_data_path = None  # Store data_path for use in signal handler
        # Layer references - 4 image layers (2 views x 2 projections)
        self.projection_yx_a_layer = None
        self.projection_yx_b_layer = None
        self.projection_zy_a_layer = None
        self.projection_zy_b_layer = None
        # Shared Shapes layers
        self.shapes_yx_layer = None
        self.shapes_zy_layer = None
        # Flag to prevent recursive shape updates
        self._syncing_shapes = False
        self.setup_ui()
        # Connect to remote client signals if available
        if remote_client is not None:
            remote_client.command_response.connect(self._on_remote_command_response)
            remote_client.progress.connect(self._on_remote_progress)

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Data Path selection
        path_group = QGroupBox("Data Path")
        path_layout = QFormLayout()

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select μManager acquisition folder...")
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_data_path)

        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit)
        path_row.addWidget(self.browse_button)

        path_layout.addRow("Data Path:", path_row)

        # Channel selection (1-indexed)
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, 20)
        self.channel_spin.setValue(1)
        path_layout.addRow("Channel:", self.channel_spin)

        # Multi-Position checkbox
        self.multi_pos_checkbox = QCheckBox("Multi-Position")
        self.multi_pos_checkbox.setToolTip("Enable for multi-position acquisitions")
        self.multi_pos_checkbox.toggled.connect(self._on_multi_pos_toggled)
        path_layout.addRow(self.multi_pos_checkbox)

        # Time selection
        self.time_spin = QSpinBox()
        self.time_spin.setRange(0, 1000)
        self.time_spin.setValue(0)
        self.time_spin.setToolTip("Timepoint to load")
        path_layout.addRow("Time:", self.time_spin)

        # Position selection (hidden by default)
        self.position_spin = QSpinBox()
        self.position_spin.setRange(0, 100)
        self.position_spin.setValue(0)
        self.position_spin.setToolTip("Position index for multi-position acquisitions")
        self.position_spin.setVisible(False)
        path_layout.addRow("Position:", self.position_spin)

        # Projection type
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["max", "sum", "mean"])
        self.projection_combo.setCurrentText("max")
        path_layout.addRow("Projection:", self.projection_combo)

        # Load button
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.load_projections)
        self.load_button.setEnabled(False)
        path_layout.addRow(self.load_button)

        # Save button - appears below Load, enabled only after data is loaded
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_bbox)
        self.save_button.setEnabled(False)
        path_layout.addRow(self.save_button)

        path_group.setLayout(path_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("Select data path and click Load")

        # ROI info display
        self.roi_info_label = QLabel("")
        self.roi_info_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(path_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.roi_info_label)
        self.setLayout(layout)

        # Connect path validation
        self.path_edit.textChanged.connect(self.validate_path)

    def browse_data_path(self):
        """Open file dialog to select data path.

        Uses SFTP browser when remote connection is active, otherwise local dialog.
        """
        if self._is_remote_connected():
            self._browse_remote()
        else:
            self._browse_local()

    def _browse_local(self):
        """Open local file dialog to select data path."""
        path = QFileDialog.getExistingDirectory(
            self, "Select μManager Acquisition Folder"
        )
        if path:
            self._remote_mode = False
            self.path_edit.setText(path)

    def _browse_remote(self):
        """Open SFTP browser dialog to select remote data path."""
        if not self.remote_client:
            return
        try:
            sftp = self.remote_client.get_sftp_client()
            # Determine home directory as initial path
            try:
                home = sftp.normalize(".")
            except Exception:
                home = "/"
            dialog = SftpBrowserDialog(
                sftp,
                parent=self,
                initial_path=home,
                title="Select Remote μManager Acquisition Folder",
            )
            if dialog.exec_() and dialog.selected_path:  # type: ignore[attr-defined]
                self._remote_mode = True
                self.path_edit.setText(dialog.selected_path)
        except Exception as e:
            QMessageBox.critical(self, "SFTP Error", f"Failed to browse remote:\n{e}")

    def validate_path(self):
        """Validate the selected data path."""
        path = self.path_edit.text()
        if self._remote_mode:
            # For remote paths, just check non-empty
            self.load_button.setEnabled(bool(path))
        else:
            self.load_button.setEnabled(bool(path and os.path.exists(path)))

    def _on_multi_pos_toggled(self, checked: bool):
        """Handle Multi-Position checkbox toggled."""
        self.position_spin.setVisible(checked)
        if checked:
            path = self.path_edit.text()
            if not path:
                return
            if self._remote_mode and self.remote_client:
                # Query position count on the remote server
                self._query_positions_remote(path)
            elif os.path.exists(path):
                # Local auto-detect
                try:
                    from pyspim.data import dispim as data
                    with data.uManagerAcquisition(path, True, np) as acq:
                        num_pos = acq.num_positions
                        self.position_spin.setRange(0, max(0, num_pos - 1))
                except Exception:
                    # If auto-detection fails, keep default range
                    pass

    def _query_positions_remote(self, data_path: str):
        """Query number of positions from remote server via signal-based pattern."""
        if not self.remote_client:
            return
        self._pending_command_type = "query_positions"
        try:
            self.remote_client.send_command(
                command="query_positions",
                params={"data_path": data_path},
                callback=None,  # Use command_response signal instead
            )
        except Exception:
            self._pending_command_type = None

    def _remove_existing_layers(self):
        """Remove any existing projection and shapes layers."""
        layers_to_remove = []
        if self.projection_yx_a_layer:
            layers_to_remove.append(self.projection_yx_a_layer)
        if self.projection_yx_b_layer:
            layers_to_remove.append(self.projection_yx_b_layer)
        if self.projection_zy_a_layer:
            layers_to_remove.append(self.projection_zy_a_layer)
        if self.projection_zy_b_layer:
            layers_to_remove.append(self.projection_zy_b_layer)
        if self.shapes_yx_layer:
            layers_to_remove.append(self.shapes_yx_layer)
        if self.shapes_zy_layer:
            layers_to_remove.append(self.shapes_zy_layer)

        for layer in layers_to_remove:
            try:
                self.viewer.layers.remove(layer)
            except (ValueError, TypeError):
                pass

        # Reset references
        self.projection_yx_a_layer = None
        self.projection_yx_b_layer = None
        self.projection_zy_a_layer = None
        self.projection_zy_b_layer = None
        self.shapes_yx_layer = None
        self.shapes_zy_layer = None

    def load_projections(self):
        """Load data and compute projections for both views.

        Uses remote server when connection is active and path is remote,
        otherwise falls back to local execution.
        """
        data_path = self.path_edit.text()
        channel = self.channel_spin.value() - 1  # Convert to 0-indexed
        projection_type = self.projection_combo.currentText()
        multi_pos = self.multi_pos_checkbox.isChecked()
        time = self.time_spin.value()
        position = self.position_spin.value()

        if not data_path:
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

        # For local mode, validate path exists
        if not self._remote_mode and not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

        # Remove any existing layers
        self._remove_existing_layers()

        self.load_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Loading data and computing projections...")
        self.roi_info_label.setText("")

        # Store current settings
        self.data_path = data_path

        if self._remote_mode and self.remote_client:
            # Remote execution
            self._load_projections_remote(
                data_path, channel, projection_type, multi_pos, time, position
            )
        else:
            # Local execution
            self._load_projections_local(
                data_path, channel, projection_type, multi_pos, time, position
            )

    def _load_projections_local(
        self, data_path, channel, projection_type, multi_pos, time, position
    ):
        """Load projections locally using ProjectionLoaderWorker."""
        self.loader_worker = ProjectionLoaderWorker(
            data_path, channel, projection_type, multi_pos, time, position
        )
        self.loader_worker.projections_loaded.connect(self.on_projections_loaded)
        self.loader_worker.error_occurred.connect(self.on_error)
        self.loader_worker.start()

    def _load_projections_remote(
        self, data_path, channel, projection_type, multi_pos, time, position
    ):
        """Load projections via remote server using signal-based pattern."""
        if not self.remote_client:
            self.on_error("Remote client not available")
            return

        params = {
            "data_path": data_path,
            "channel": channel,
            "projection_type": projection_type,
            "multi_pos": multi_pos,
            "time": time,
            "position": position,
        }

        # Store context for signal handler
        self._pending_command_type = "compute_projections"
        self._pending_data_path = data_path

        try:
            # Use callback=None — rely on command_response signal (marshaled to main thread)
            self.remote_client.send_command(
                command="compute_projections",
                params=params,
                callback=None,
                progress_callback=None,
            )
        except Exception as e:
            self.on_error(f"Failed to send command: {e}")

    def _on_remote_command_response(self, response: dict):
        """Handle command_response signal from RemoteClient (runs on main thread).

        Only handles responses for commands we sent (compute_projections, query_positions).
        Other commands (like 'compute' from test connection) are ignored.
        """
        if self._pending_command_type == "compute_projections":
            self._pending_command_type = None
            if response.get("success"):
                result = response.get("result", {})
                # Convert volume_shape lists back to tuples for consistency
                if "volume_shape_a" in result and isinstance(result["volume_shape_a"], list):
                    result["volume_shape_a"] = tuple(result["volume_shape_a"])
                if "volume_shape_b" in result and isinstance(result["volume_shape_b"], list):
                    result["volume_shape_b"] = tuple(result["volume_shape_b"])
                result["data_path"] = self._pending_data_path
                self._pending_data_path = None
                self.on_projections_loaded(result)
            else:
                error_msg = response.get("error", "Unknown error")
                self._pending_data_path = None
                self.on_error(error_msg)
        elif self._pending_command_type == "query_positions":
            self._pending_command_type = None
            if response.get("success"):
                num_pos = response.get("result", {}).get("num_positions", 1)
                self.position_spin.setRange(0, max(0, num_pos - 1))

    def _on_remote_progress(self, message: str, percentage: int):
        """Handle progress signal from RemoteClient (runs on main thread)."""
        self.status_label.setText(message)
        if percentage >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)

    def _is_remote_connected(self):
        """Check if remote connection is active."""
        return self.remote_client is not None and self.remote_client.is_connected

    def on_projections_loaded(self, result):
        """Handle successful projection loading - schedule on main thread."""
        QTimer.singleShot(0, lambda: self._on_projections_loaded_main(result))

    def _on_projections_loaded_main(self, result):
        """Add layers on main thread."""
        try:
            yx_proj_a = result["yx_proj_a"]  # original shape (Y, X)
            zy_proj_a = result["zy_proj_a"]  # original shape (Z, Y)
            yx_proj_b = result["yx_proj_b"]  # original shape (Y, X)
            zy_proj_b = result["zy_proj_b"]  # original shape (Z, Y)
            volume_shape_a = result["volume_shape_a"]
            volume_shape_b = result["volume_shape_b"]

            # Transpose YX from (Y, X) to (X, Y) so Y is horizontal in both views
            yx_proj_a_t = yx_proj_a.T  # shape: (X, Y)
            yx_proj_b_t = yx_proj_b.T  # shape: (X, Y)


            # Determine shapes from View A (assumed same size as View B)
            X, Y = yx_proj_a_t.shape
            Z, Y_zy = zy_proj_a.shape

            # Add YX projection layer View A (transposed: X, Y)
            self.projection_yx_a_layer = self.viewer.add_image(
                yx_proj_a_t,
                name="YX Projection (View A)",
                colormap="red",
                opacity=0.8,
                blending="additive",
            )

            # Add YX projection layer View B (transposed: X, Y) - same coordinates as A
            self.projection_yx_b_layer = self.viewer.add_image(
                yx_proj_b_t,
                name="YX Projection (View B)",
                colormap="cyan",
                opacity=0.8,
                blending="additive",
            )

            # Add ZY projection layer View A (Z, Y) - offset horizontally to not overlap YX
            offset_y = -Z  # gap between YX and ZY views
            self.projection_zy_a_layer = self.viewer.add_image(
                zy_proj_a,
                name="ZY Projection (View A)",
                colormap="red",
                opacity=0.8,
                blending="additive",
                translate=(offset_y, 0),
            )

            # Add ZY projection layer View B (Z, Y) - same offset as A
            self.projection_zy_b_layer = self.viewer.add_image(
                zy_proj_b,
                name="ZY Projection (View B)",
                colormap="cyan",
                opacity=0.8,
                blending="additive",
                translate=(offset_y, 0),
            )

            # Add shared Shapes layers with full-coverage rectangles
            # YX projection (transposed): shape (X, Y), coords are (row=X, col=Y)
            yx_rect = [[(0, 0), (0, Y-1), (X-1, Y-1), (X-1, 0)]]
            self.shapes_yx_layer = self.viewer.add_shapes(
                yx_rect,
                shape_type="rectangle",
                name="YX ROI (Shared)",
                edge_color="yellow",
                face_color="transparent",
                edge_width=2,
                translate=self.projection_yx_a_layer.translate,
            )
            self.shapes_yx_layer.editable = True

            # ZY projection: shape (Z, Y), coords are (row=Z, col=Y)
            zy_rect = [[(0, 0), (0, Y_zy - 1), (Z - 1, Y_zy - 1), (Z - 1, 0)]]
            self.shapes_zy_layer = self.viewer.add_shapes(
                zy_rect,
                shape_type="rectangle",
                name="ZY ROI (Shared)",
                edge_color="yellow",
                face_color="transparent",
                edge_width=2,
                translate=self.projection_zy_a_layer.translate,
            )
            self.shapes_zy_layer.editable = True

            # Connect shapes data change events for Y-axis sync
            self.shapes_yx_layer.events.data.connect(self.on_yx_shapes_changed)
            self.shapes_zy_layer.events.data.connect(self.on_zy_shapes_changed)

            # Update UI state
            self.load_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText("Projections loaded!")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.on_error(str(e))

    def _commit_shape_edits(self, layer):
        """Force a shapes layer to commit any pending interactive edits.

        When shapes are edited interactively in transform/direct mode,
        napari keeps the edits in an internal transform buffer until the
        mode switches away from editing. Calling this method forces the
        layer to finalize pending edits by temporarily switching to
        pan_zoom mode and back, which triggers the data event and commits
        the changes to layer.data.
        """
        if layer.mode in ('transform', 'direct'):
            # Store the current mode to restore it after
            previous_mode = layer.mode
            # Switch to pan_zoom to finalize pending edits
            layer.mode = 'pan_zoom'
            # Switch back to the previous mode
            layer.mode = previous_mode

    def on_yx_shapes_changed(self):
        """Handle changes to YX shapes layer - sync Y bounds to ZY layer.

        YX layer is transposed (X, Y): row=X, col=Y
        ZY layer is (Z, Y): row=Z, col=Y
        Y is on the column axis for both views.
        """
        if self._syncing_shapes:
            return
        self._syncing_shapes = True

        try:
            if not self.shapes_yx_layer or not self.shapes_zy_layer:
                return
            if len(self.shapes_yx_layer.data) == 0:
                return

            # Extract Y bounds from YX shapes (column coordinates, since YX is transposed)
            yx_corners = list(self.shapes_yx_layer.data[0])
            y_min = int(min(c[1] for c in yx_corners))
            y_max = int(max(c[1] for c in yx_corners))

            # Update ZY shapes with new Y bounds (column coordinates in ZY view)
            zy_corners = list(self.shapes_zy_layer.data[0])
            zy_y_min = min(c[1] for c in zy_corners)
            zy_y_max = max(c[1] for c in zy_corners)

            new_zy_corners = []
            for corner in zy_corners:
                z, y = corner
                if y == zy_y_min:
                    new_y = y_min
                else:
                    new_y = y_max
                new_zy_corners.append([z, new_y])

            self.shapes_zy_layer.data = [new_zy_corners]

        finally:
            self._syncing_shapes = False

    def on_zy_shapes_changed(self):
        """Handle changes to ZY shapes layer - sync Y bounds to YX layer.

        ZY layer is (Z, Y): row=Z, col=Y
        YX layer is transposed (X, Y): row=X, col=Y
        Y is on the column axis for both views.
        """
        if self._syncing_shapes:
            return
        self._syncing_shapes = True

        try:
            if not self.shapes_yx_layer or not self.shapes_zy_layer:
                return
            if len(self.shapes_zy_layer.data) == 0:
                return

            # Extract Y bounds from ZY shapes (column coordinates)
            zy_corners = list(self.shapes_zy_layer.data[0])
            y_min = int(min(c[1] for c in zy_corners))
            y_max = int(max(c[1] for c in zy_corners))

            # Update YX shapes with new Y bounds (column coordinates in transposed YX view)
            yx_corners = list(self.shapes_yx_layer.data[0])
            yx_y_min = min(c[1] for c in yx_corners)
            yx_y_max = max(c[1] for c in yx_corners)

            new_yx_corners = []
            for corner in yx_corners:
                x, y = corner
                if y == yx_y_min:
                    new_y = y_min
                else:
                    new_y = y_max
                new_yx_corners.append([x, new_y])

            self.shapes_yx_layer.data = [new_yx_corners]

        finally:
            self._syncing_shapes = False

    def _update_roi_info(self, x_start, x_end, y_start, y_end, z_start, z_end):
        """Update ROI information display."""
        info_text = f"""
        <b>ROI Bounding Box:</b><br>
        Z: {z_start} to {z_end} ({z_end - z_start} pixels)<br>
        Y: {y_start} to {y_end} ({y_end - y_start} pixels)<br>
        X: {x_start} to {x_end} ({x_end - x_start} pixels)
        """
        self.roi_info_label.setText(info_text)

    def save_bbox(self):
        """Save bounding box to JSON file.

        Uses SFTP when in remote mode, otherwise saves locally.
        """
        if not self.shapes_yx_layer or not self.shapes_zy_layer:
            QMessageBox.warning(self, "Error", "Please load projections first")
            return
        if not self.data_path:
            QMessageBox.warning(self, "Error", "No data path set")
            return

        # Commit any pending interactive edits before reading layer.data
        # This is necessary because napari keeps transform edits in an
        # internal buffer until the layer mode switches away from editing
        self._commit_shape_edits(self.shapes_yx_layer)
        self._commit_shape_edits(self.shapes_zy_layer)

        # Extract bounding box from YX Shapes layer
        # YX is transposed (X, Y): row=X, col=Y
        yx_corners = list(self.shapes_yx_layer.data[0])
        x_start = int(min(c[0] for c in yx_corners))
        x_end = int(max(c[0] for c in yx_corners)) + 1
        y_start = int(min(c[1] for c in yx_corners))
        y_end = int(max(c[1] for c in yx_corners)) + 1

        # Extract bounding box from ZY Shapes layer
        # ZY: (row=Z, col=Y)
        zy_corners = list(self.shapes_zy_layer.data[0])
        z_start = int(min(c[0] for c in zy_corners))
        z_end = int(max(c[0] for c in zy_corners)) + 1

        # Reconstruct 3D bbox: ((z_start, z_end), (y_start, y_end), (x_start, x_end))
        bbox_3d = [[z_start, z_end], [y_start, y_end], [x_start, x_end]]

        # Save to JSON file (shared for both views)
        filename = "bbox_raw.json"

        if self._remote_mode and self.remote_client:
            self._save_bbox_remote(bbox_3d, filename)
        else:
            self._save_bbox_local(bbox_3d, filename)

    def _save_bbox_local(self, bbox_3d, filename):
        """Save bounding box to local JSON file."""
        assert self.data_path is not None
        output_path = os.path.join(self.data_path, filename)
        try:
            with open(output_path, "w") as f:
                json.dump(bbox_3d, f)
            self._update_roi_info_from_bbox(bbox_3d)
            self.status_label.setText("Bounding box saved locally.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save bounding box: {e}")

    def _save_bbox_remote(self, bbox_3d, filename):
        """Save bounding box to remote server via SFTP."""
        if not self.remote_client:
            return
        try:
            sftp = self.remote_client.get_sftp_client()
            remote_path = f"{self.data_path}/{filename}"
            json_content = json.dumps(bbox_3d)
            with sftp.open(remote_path, "w") as f:
                f.write(json_content)
            self._update_roi_info_from_bbox(bbox_3d)
            self.status_label.setText("Bounding box saved to remote server.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save bounding box remotely: {e}")

    def _update_roi_info_from_bbox(self, bbox_3d):
        """Update ROI info display from bbox_3d list."""
        z_start, z_end = bbox_3d[0]
        y_start, y_end = bbox_3d[1]
        x_start, x_end = bbox_3d[2]
        self._update_roi_info(x_start, x_end, y_start, y_end, z_start, z_end)

    def on_error(self, error_msg):
        """Handle error."""
        QMessageBox.critical(self, "Error", f"Operation failed: {error_msg}")
        self.status_label.setText("Error during operation")
        self.load_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def set_input_data(self, data_dict):
        """Set input data from previous step (for backward compatibility)."""
        # No longer used - we load data directly now
        pass
