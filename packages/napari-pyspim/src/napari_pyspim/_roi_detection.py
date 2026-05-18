"""
ROI detection widget for manual region of interest detection and cropping.
"""

import json
import os

import numpy as np
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread, QTimer
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


class ProjectionLoaderWorker(QThread):
    """Worker thread for loading data and computing projections for both views."""

    projections_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

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

    roi_applied = pyqtSignal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.loader_worker = None
        self.data_path = None
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

        path_group.setLayout(path_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("Select data path and click Load")

        # ROI info display
        self.roi_info_label = QLabel("")
        self.roi_info_label.setWordWrap(True)

        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_bbox)
        self.save_button.setEnabled(False)

        # Add widgets to layout
        layout.addWidget(path_group)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.roi_info_label)
        layout.addWidget(self.save_button)
        layout.addStretch()

        self.setLayout(layout)

        # Connect path validation
        self.path_edit.textChanged.connect(self.validate_path)

    def browse_data_path(self):
        """Open file dialog to select data path."""
        path = QFileDialog.getExistingDirectory(
            self, "Select μManager Acquisition Folder"
        )
        if path:
            self.path_edit.setText(path)

    def validate_path(self):
        """Validate the selected data path."""
        path = self.path_edit.text()
        if path and os.path.exists(path):
            self.load_button.setEnabled(True)
        else:
            self.load_button.setEnabled(False)

    def _on_multi_pos_toggled(self, checked: bool):
        """Handle Multi-Position checkbox toggled."""
        self.position_spin.setVisible(checked)
        if checked:
            # Try to auto-detect number of positions from the data
            path = self.path_edit.text()
            if path and os.path.exists(path):
                try:
                    from pyspim.data import dispim as data
                    with data.uManagerAcquisition(path, True, np) as acq:
                        num_pos = acq.num_positions
                        self.position_spin.setRange(0, max(0, num_pos - 1))
                except Exception:
                    # If auto-detection fails, keep default range
                    pass

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
        """Load data and compute projections for both views."""
        data_path = self.path_edit.text()
        channel = self.channel_spin.value() - 1  # Convert to 0-indexed
        projection_type = self.projection_combo.currentText()
        multi_pos = self.multi_pos_checkbox.isChecked()
        time = self.time_spin.value()
        position = self.position_spin.value()

        if not data_path or not os.path.exists(data_path):
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

        # Create and start worker (loads both views)
        self.loader_worker = ProjectionLoaderWorker(
            data_path, channel, projection_type, multi_pos, time, position
        )
        self.loader_worker.projections_loaded.connect(self.on_projections_loaded)
        self.loader_worker.error_occurred.connect(self.on_error)
        self.loader_worker.start()

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
                opacity=0.5,
                blending="additive",
            )

            # Add YX projection layer View B (transposed: X, Y) - same coordinates as A
            self.projection_yx_b_layer = self.viewer.add_image(
                yx_proj_b_t,
                name="YX Projection (View B)",
                colormap="cyan",
                opacity=0.5,
                blending="additive",
            )

            # Add ZY projection layer View A (Z, Y) - offset horizontally to not overlap YX
            offset_y = -Z  # gap between YX and ZY views
            self.projection_zy_a_layer = self.viewer.add_image(
                zy_proj_a,
                name="ZY Projection (View A)",
                colormap="red",
                opacity=0.5,
                blending="additive",
                translate=(offset_y, 0),
            )

            # Add ZY projection layer View B (Z, Y) - same offset as A
            self.projection_zy_b_layer = self.viewer.add_image(
                zy_proj_b,
                name="ZY Projection (View B)",
                colormap="cyan",
                opacity=0.5,
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
        """Save bounding box to JSON file."""
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
        output_path = os.path.join(self.data_path, filename)

        try:
            with open(output_path, "w") as f:
                json.dump(bbox_3d, f)
            self._update_roi_info(x_start, x_end, y_start, y_end, z_start, z_end)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save bounding box: {e}")

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
