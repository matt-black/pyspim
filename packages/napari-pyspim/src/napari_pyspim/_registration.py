"""
Registration widget for aligning dual-view SPIM data.

This widget combines data loading, deskewing, projection-based manual alignment,
and automated registration into a single workflow.
"""

import json
import math
import os

try:
    import cupy as cp
except ImportError:
    import numpy as cp
    print("Warning: CuPy not available, using NumPy (CPU only)")
import numpy as np
from napari.utils.transforms import Affine
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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

# Lazy imports to avoid CUDA compilation at module level
# from pyspim.reg import pcc, powell
# from pyspim.interp import affine
# from pyspim.util import pad_to_same_size, launch_params_for_volume

class LoadDeskewWorker(QThread):
    """Worker thread for loading data, deskewing, and computing projections."""

    ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(
        self,
        data_path: str,
        channel: int,
        projection_type: str,
        pixel_size: float,
        step_size: float,
        theta: float,
        method: str = "orthogonal",
        ignore_bbox: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.channel = channel  # 0-indexed
        self.projection_type = projection_type
        self.pixel_size = pixel_size
        self.step_size = step_size
        self.theta = theta
        self.method = method
        self.ignore_bbox = ignore_bbox

    def _load_bbox(self):
        """Load bounding box from bbox_raw.json if it exists."""
        if self.ignore_bbox:
            self.progress_updated.emit("Ignoring bounding box - loading full dataset")
            return None
        bbox_path = os.path.join(self.data_path, "bbox_raw.json")
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, "r") as f:
                    bbox = json.load(f)
                # bbox format: [[z_start, z_end], [y_start, y_end], [x_start, x_end]]
                window = (
                    slice(bbox[0][0], bbox[0][1]),
                    slice(bbox[1][0], bbox[1][1]),
                    slice(bbox[2][0], bbox[2][1]),
                )
                self.progress_updated.emit(
                    f"Loading data subset from bbox_raw.json: {bbox}"
                )
                return window
            except (json.JSONDecodeError, IndexError, TypeError) as e:
                self.progress_updated.emit(
                    f"Warning: Could not parse bbox_raw.json ({e}), loading full dataset"
                )
        return None

    def _compute_projections(self, volume, proj_type):
        """Compute YX, ZY, and XZ projections from a volume."""
        if proj_type == "max":
            yx_proj = np.max(volume, axis=0)  # shape: (Y, X)
            zy_proj = np.max(volume, axis=2)  # shape: (Z, Y)
            xz_proj = np.max(volume, axis=1)  # shape: (Z, X)
        elif proj_type == "sum":
            yx_proj = np.sum(volume, axis=0)
            zy_proj = np.sum(volume, axis=2)
            xz_proj = np.sum(volume, axis=1)
        elif proj_type == "mean":
            yx_proj = np.mean(volume, axis=0)
            zy_proj = np.mean(volume, axis=2)
            xz_proj = np.mean(volume, axis=1)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        return yx_proj, zy_proj, xz_proj

    def run(self):
        """Load data, deskew, and compute projections in background thread."""
        try:
            from pyspim.data import dispim as data
            from pyspim import deskew as dsk

            # Load bounding box if present
            window = self._load_bbox()
            print(window)

            # Load data
            self.progress_updated.emit("Loading data...")
            with data.uManagerAcquisition(self.data_path, False, np) as acq:
                volume_a = acq.get("a", self.channel, 0, window=window)
                volume_b = acq.get("b", self.channel, 0, window=window)

            self.progress_updated.emit(
                f"Data loaded - A: {volume_a.shape}, B: {volume_b.shape}"
            )

            # Calculate lateral step size
            step_size_lat = self.step_size / math.cos(self.theta)

            # Deskew View A (direction = 1)
            self.progress_updated.emit("Deskewing View A...")
            if self.method == "shear":
                kwargs = {
                    "rotation_thetas": (0, 0, 0),
                    "interp_method": "linear",
                    "auto_crop": False,
                    "preserve_dtype": False,
                    "block_size": (8,8,8),
                }
            elif self.method == "ortho":
                kwargs = {
                    "preserve_dtype": False,
                    "stream": None,
                }
            else:
                kwargs = {}
            a_dsk = dsk.deskew_stage_scan(
                volume_a, self.pixel_size, step_size_lat, 1, 
                theta=(self.theta / (math.pi/180.0)),
                method=self.method,
                **kwargs,
            )
            try:
                a_dsk = a_dsk.get()
            except:
                pass
            # Deskew View B (direction = -1)
            self.progress_updated.emit("Deskewing View B...")
            b_dsk = dsk.deskew_stage_scan(
                volume_b, self.pixel_size, step_size_lat, -1, 
                theta=(self.theta / (math.pi/180.0)),
                method=self.method,
                **kwargs,
            )
            try:
                b_dsk = b_dsk.get()
            except:
                pass
            a_dsk = a_dsk.astype(np.float32)
            b_dsk = b_dsk.astype(np.float32)

            self.progress_updated.emit(f"Deskewing completed")

            # Compute projections
            self.progress_updated.emit("Computing projections...")
            yx_proj_a, zy_proj_a, xz_proj_a = self._compute_projections(a_dsk, self.projection_type)
            yx_proj_b, zy_proj_b, xz_proj_b = self._compute_projections(b_dsk, self.projection_type)

            result = {
                "a_deskewed": a_dsk,
                "b_deskewed": b_dsk,
                "yx_proj_a": yx_proj_a,
                "zy_proj_a": zy_proj_a,
                "xz_proj_a": xz_proj_a,
                "yx_proj_b": yx_proj_b,
                "zy_proj_b": zy_proj_b,
                "xz_proj_b": xz_proj_b,
                "volume_shape_a": a_dsk.shape,
                "volume_shape_b": b_dsk.shape,
                "method": self.method,
                "step_size_lat": step_size_lat,
            }

            self.ready.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class RegistrationWorker(QThread):
    """Worker thread for registration."""

    registered = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(
        self,
        a_deskewed,
        b_deskewed,
        transform_type="t+r+s",
        initial_translation=None,
    ):
        super().__init__()
        self.a_deskewed = a_deskewed
        self.b_deskewed = b_deskewed
        self.transform_type = transform_type
        self.initial_translation = initial_translation if initial_translation is not None else [0, 0, 0]

    def run(self):
        """Perform registration in background thread."""
        try:
            # Lazy imports to avoid CUDA compilation at module level
            from pyspim.interp import affine
            from pyspim.reg import powell
            from pyspim.util import launch_params_for_volume, pad_to_same_size

            self.progress_updated.emit("Padding volumes to same size...")

            # Pad volumes to same size
            a_dsk, b_dsk = pad_to_same_size(self.a_deskewed, self.b_deskewed)

            # Set up initial parameters and bounds
            self.progress_updated.emit("Setting up optimization parameters...")

            # Use initial translation from pre-reg transform (in pixel units), or default to [0, 0, 0]
            t0 = self.initial_translation

            if self.transform_type == "t":
                par0 = t0
                bounds = [(t - 20, t + 20) for t in t0]
            elif self.transform_type == "t+r":
                par0 = np.concatenate([t0, np.asarray([0, 0, 0])])
                bounds = [(t - 20, t + 20) for t in t0] + [(-5, 5)] * 3
            elif self.transform_type == "t+r+s":
                par0 = np.concatenate(
                    [t0, np.asarray([0, 0, 0]), np.asarray([1, 1, 1])]
                )
                bounds = (
                    [(t - 20, t + 20) for t in t0] + [(-5, 5)] * 3 + [(0.9, 1.1)] * 3
                )

            # Determine launch parameters for GPU
            self.progress_updated.emit("Optimizing registration...")
            launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)

            # Perform optimization
            T, res = powell.optimize_affine_piecewise(
                cp.asarray(a_dsk),
                cp.asarray(b_dsk),
                metric="cr",
                transform=self.transform_type,
                interp_method="cubspl",
                par0=par0,
                bounds=bounds,
                kernel_launch_params=launch_par,
                verbose=False,
            )

            # Apply transformation
            self.progress_updated.emit("Applying transformation...")
            b_reg = affine.transform(
                cp.asarray(b_dsk),
                T,
                interp_method="cubspl",
                preserve_dtype=True,
                out_shp=None,
                block_size_z=8,
                block_size_y=8,
                block_size_x=8,
            ).get()

            # Crop to smallest size
            min_sze = [min(a, b) for a, b in zip(a_dsk.shape, b_reg.shape)]
            a_final = a_dsk[: min_sze[0], : min_sze[1], : min_sze[2]]
            b_final = b_reg[: min_sze[0], : min_sze[1], : min_sze[2]]

            # Calculate correlation ratio
            cr = 1 - res.fun

            result = {
                "a_registered": a_final,
                "b_registered": b_final,
                "transform_matrix": T,
                "correlation_ratio": cr,
                "transform_type": self.transform_type,
            }

            self.registered.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class RegistrationWidget(QWidget):
    """Widget for registering dual-view data.
    
    Combines data loading, deskewing, projection-based manual alignment,
    and automated registration.
    """

    registered = pyqtSignal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.load_worker = None
        self.reg_worker = None
        self.input_data = None
        # Stored deskewed data for registration
        self.a_deskewed = None
        self.b_deskewed = None
        # Layer references for projection display
        self.projection_yx_a_layer = None
        self.projection_yx_b_layer = None
        self.projection_zy_a_layer = None
        self.projection_zy_b_layer = None
        self.projection_xz_a_layer = None
        self.projection_xz_b_layer = None
        # Pre-registration transform state (stored in micrometers: [tz, ty, tx])
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        # Flag to prevent recursive sync when programmatically updating layers
        self._syncing_transforms = False
        # Store pixel size for layer scale
        self._pixel_size = None
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

        # Projection type
        self.projection_combo = QComboBox()
        self.projection_combo.addItems(["max", "sum", "mean"])
        self.projection_combo.setCurrentText("max")
        path_layout.addRow("Projection:", self.projection_combo)

        self.ignore_bbox_checkbox = QCheckBox("Ignore bounding box")
        self.ignore_bbox_checkbox.setToolTip("If checked, load all data for deskewing instead of using bbox_raw.json")
        path_layout.addRow(self.ignore_bbox_checkbox)

        path_group.setLayout(path_layout)

        # Deskewing parameters
        deskew_group = QGroupBox("Deskewing Parameters")
        deskew_layout = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(["orthogonal", "dispim", "shear"])
        self.method_combo.setCurrentText("orthogonal")

        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.1, 10.0)
        self.step_size_spin.setValue(0.5)
        self.step_size_spin.setSuffix(" μm")
        self.step_size_spin.setDecimals(3)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 1.0)
        self.pixel_size_spin.setDecimals(4)
        self.pixel_size_spin.setValue(0.1625)
        self.pixel_size_spin.setSuffix(" μm")
        
        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0, 90)
        self.theta_spin.setValue(45)
        self.theta_spin.setSuffix("°")
        self.theta_spin.setDecimals(1)

        deskew_layout.addRow("Method:", self.method_combo)
        deskew_layout.addRow("Step Size:", self.step_size_spin)
        deskew_layout.addRow("Pixel Size:", self.pixel_size_spin)
        deskew_layout.addRow("Theta (angle):", self.theta_spin)
        deskew_group.setLayout(deskew_layout)

        # Load & Deskew button
        self.load_deskew_button = QPushButton("Load + Deskew")
        self.load_deskew_button.clicked.connect(self.load_and_deskew)
        self.load_deskew_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("Select data path and click Load + Deskew")

        # Pre-Registration group
        pre_reg_group = QGroupBox("Pre-Registration")
        pre_reg_layout = QVBoxLayout()

        self.pre_reg_label = QLabel("Pre-reg transform: (0.0, 0.0, 0.0)")
        pre_reg_layout.addWidget(self.pre_reg_label)

        self.reset_transform_button = QPushButton("Reset Transform")
        self.reset_transform_button.setEnabled(False)
        self.reset_transform_button.clicked.connect(self._reset_transform)
        pre_reg_layout.addWidget(self.reset_transform_button)

        pre_reg_tip = QLabel("Use transform mode on View B layers to pre-align.")
        pre_reg_tip.setWordWrap(True)
        pre_reg_layout.addWidget(pre_reg_tip)

        pre_reg_group.setLayout(pre_reg_layout)

        # Registration parameters
        params_group = QGroupBox("Registration Parameters")
        params_layout = QFormLayout()

        self.transform_combo = QComboBox()
        self.transform_combo.addItems(["t", "t+r", "t+r+s"])
        self.transform_combo.setCurrentText("t+r+s")

        params_layout.addRow("Transform Type:", self.transform_combo)

        self.crop_to_common_checkbox = QCheckBox("Crop to Common")
        self.crop_to_common_checkbox.setToolTip(
            "When enabled, crops deskewed volumes to the overlapping region "
            "determined by the pre-reg translation before registration."
        )
        params_layout.addRow(self.crop_to_common_checkbox)

        params_group.setLayout(params_layout)

        # Register button
        self.register_button = QPushButton("Register Data")
        self.register_button.clicked.connect(self.register_data)
        self.register_button.setEnabled(False)

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(path_group)
        layout.addWidget(deskew_group)
        layout.addWidget(self.load_deskew_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(pre_reg_group)
        layout.addWidget(params_group)
        layout.addWidget(self.register_button)
        layout.addWidget(self.results_label)
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
            self.load_deskew_button.setEnabled(True)
        else:
            self.load_deskew_button.setEnabled(False)

    def _remove_existing_layers(self):
        """Remove any existing projection layers."""
        layers_to_remove = []
        if self.projection_yx_a_layer:
            layers_to_remove.append(self.projection_yx_a_layer)
        if self.projection_yx_b_layer:
            layers_to_remove.append(self.projection_yx_b_layer)
        if self.projection_zy_a_layer:
            layers_to_remove.append(self.projection_zy_a_layer)
        if self.projection_zy_b_layer:
            layers_to_remove.append(self.projection_zy_b_layer)
        if self.projection_xz_a_layer:
            layers_to_remove.append(self.projection_xz_a_layer)
        if self.projection_xz_b_layer:
            layers_to_remove.append(self.projection_xz_b_layer)

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
        self.projection_xz_a_layer = None
        self.projection_xz_b_layer = None

        # Reset pre-reg transform state
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        # Reset crop checkbox
        self.crop_to_common_checkbox.setChecked(False)
        self._syncing_transforms = False
        self._pixel_size = None
        self.reset_transform_button.setEnabled(False)
        self._update_pre_reg_label()

    def load_and_deskew(self):
        """Load data, deskew, and display projections."""
        data_path = self.path_edit.text()
        channel = self.channel_spin.value() - 1  # Convert to 0-indexed
        projection_type = self.projection_combo.currentText()

        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

        # Remove any existing layers
        self._remove_existing_layers()

        # Reset stored data
        self.a_deskewed = None
        self.b_deskewed = None

        self.load_deskew_button.setEnabled(False)
        self.register_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Loading data and deskewing...")
        self.results_label.setText("")

        # Get deskewing parameters
        pixel_size = self.pixel_size_spin.value()
        step_size = self.step_size_spin.value()
        theta_deg = self.theta_spin.value()
        theta_rad = theta_deg * math.pi / 180
        method = self.method_combo.currentText()
        

        # Create and start worker
        ignore_bbox = self.ignore_bbox_checkbox.isChecked()
        self.load_worker = LoadDeskewWorker(
            data_path, channel, projection_type,
            pixel_size, step_size, theta_rad,
            method, ignore_bbox,
        )
        self.load_worker.ready.connect(self.on_load_deskew_ready)
        self.load_worker.error_occurred.connect(self.on_error)
        self.load_worker.progress_updated.connect(self.update_progress)
        self.load_worker.start()

    def on_load_deskew_ready(self, result):
        """Handle successful load and deskew - schedule layer creation on main thread."""
        QTimer.singleShot(0, lambda: self._on_load_deskew_ready_main(result))

    def _on_load_deskew_ready_main(self, result):
        """Add projection layers on main thread."""
        try:
            # Store deskewed data for registration
            self.a_deskewed = result["a_deskewed"]
            self.b_deskewed = result["b_deskewed"]

            # Get projections
            yx_proj_a = result["yx_proj_a"]  # original shape (Y, X)
            zy_proj_a = result["zy_proj_a"]  # original shape (Z, Y)
            xz_proj_a = result["xz_proj_a"]  # original shape (Z, X)
            yx_proj_b = result["yx_proj_b"]  # original shape (Y, X)
            zy_proj_b = result["zy_proj_b"]  # original shape (Z, Y)
            xz_proj_b = result["xz_proj_b"]  # original shape (Z, X)
            volume_shape_a = result["volume_shape_a"]
            volume_shape_b = result["volume_shape_b"]

            # Transpose YX from (Y, X) to (X, Y) so Y is horizontal in both views
            yx_proj_a_t = yx_proj_a.T  # shape: (X, Y)
            yx_proj_b_t = yx_proj_b.T  # shape: (X, Y)

            # Determine shapes from View A
            X, Y = yx_proj_a_t.shape
            Z, Y_zy = zy_proj_a.shape

            # Get pixel size for layer scale (in micrometers)
            pixel_size = self.pixel_size_spin.value()
            self._pixel_size = pixel_size

            # Convert display offsets from pixels to micrometers
            offset_z_um = -Z * pixel_size  # gap above YX view
            offset_x_um = Y * pixel_size   # offset to right of YX view

            # Add YX projection layer View A (transposed: X, Y)
            self.projection_yx_a_layer = self.viewer.add_image(
                yx_proj_a_t,
                name="YX Projection (View A)",
                colormap="red",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
            )

            # Add YX projection layer View B (transposed: X, Y) - same coordinates as A
            self.projection_yx_b_layer = self.viewer.add_image(
                yx_proj_b_t,
                name="YX Projection (View B)",
                colormap="cyan",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
            )

            # Add ZY projection layer View A (Z, Y) - offset vertically (above) YX
            self.projection_zy_a_layer = self.viewer.add_image(
                zy_proj_a,
                name="ZY Projection (View A)",
                colormap="red",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(offset_z_um, 0),
            )

            # Add ZY projection layer View B (Z, Y) - same offset as A
            self.projection_zy_b_layer = self.viewer.add_image(
                zy_proj_b,
                name="ZY Projection (View B)",
                colormap="cyan",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(offset_z_um, 0),
            )

            # Add XZ projection layer View A (X, Z) - offset horizontally (right of) YX
            # XZ shape is (Z, X), transpose to (X, Z) for display
            xz_proj_a_t = xz_proj_a.T  # shape: (X, Z)
            self.projection_xz_a_layer = self.viewer.add_image(
                xz_proj_a_t,
                name="XZ Projection (View A)",
                colormap="red",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(0, offset_x_um),
            )

            # Add XZ projection layer View B (X, Z) - same offset as A
            xz_proj_b_t = xz_proj_b.T  # shape: (X, Z)
            self.projection_xz_b_layer = self.viewer.add_image(
                xz_proj_b_t,
                name="XZ Projection (View B)",
                colormap="cyan",
                opacity=0.5,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(0, offset_x_um),
            )

            # Connect affine change events for View B layers
            # The Transform mode in napari updates layer.affine, not layer.translate
            for layer in [self.projection_yx_b_layer, self.projection_zy_b_layer, self.projection_xz_b_layer]:
                if layer is not None:
                    layer.events.affine.connect(self._on_view_b_layer_affine_changed)

            # Lock View A layers (make non-editable)
            for layer in [self.projection_yx_a_layer, self.projection_zy_a_layer, self.projection_xz_a_layer]:
                if layer is not None:
                    layer.editable = False

            # Enable reset button
            self.reset_transform_button.setEnabled(True)

            # Update UI state
            self.load_deskew_button.setEnabled(True)
            self.register_button.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.status_label.setText(
                "Deskewing completed!"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.on_error(str(e))

    def _on_view_b_layer_affine_changed(self, event):
        """Handle affine transform change on View B projection layers.

        When the user drags a View B layer in Transform mode, napari updates
        layer.affine. Extract the translation from layer.affine.translate
        (in micrometers since layers have scale set), update the shared
        pre-reg transform, and sync other View B layers.
        """
        # Prevent recursive sync
        if self._syncing_transforms:
            return

        layer = event.source
        if layer is None:
            return

        # Extract translation from the affine transform
        # Values are in micrometers (world coordinates) since layers have scale set
        current_affine_translate = list(layer.affine.translate)

        # Determine which volume axes this layer corresponds to
        if layer is self.projection_yx_b_layer:
            # Display (X, Y) -> affine_translate[0] = tx, [1] = ty
            tx = current_affine_translate[0]
            ty = current_affine_translate[1]
            self._pre_reg_transform[2] = tx  # tx
            self._pre_reg_transform[1] = ty  # ty
        elif layer is self.projection_zy_b_layer:
            # Display (Z, Y) -> affine_translate[0] = tz, [1] = ty
            tz = current_affine_translate[0]
            ty = current_affine_translate[1]
            self._pre_reg_transform[0] = tz  # tz
            self._pre_reg_transform[1] = ty  # ty
        elif layer is self.projection_xz_b_layer:
            # Display (X, Z) -> affine_translate[0] = tx, [1] = tz
            tx = current_affine_translate[0]
            tz = current_affine_translate[1]
            self._pre_reg_transform[2] = tx  # tx
            self._pre_reg_transform[0] = tz  # tz

        # Sync other View B layers and update UI
        self._sync_view_b_layers()
        self._update_pre_reg_label()

    def _sync_view_b_layers(self):
        """Apply the current pre-reg transform to all View B layers via affine.

        Each layer gets the translation components that correspond to its
        displayed volume axes. Values are in micrometers.
        """
        self._syncing_transforms = True
        try:
            tz, ty, tx = self._pre_reg_transform

            # YX layer: display (X, Y) -> affine translate by (tx, ty)
            if self.projection_yx_b_layer is not None:
                matrix = np.eye(3)
                matrix[0, 2] = tx
                matrix[1, 2] = ty
                self.projection_yx_b_layer.affine = Affine(
                    affine_matrix=matrix,
                    ndim=2,
                )

            # ZY layer: display (Z, Y) -> affine translate by (tz, ty)
            if self.projection_zy_b_layer is not None:
                matrix = np.eye(3)
                matrix[0, 2] = tz
                matrix[1, 2] = ty
                self.projection_zy_b_layer.affine = Affine(
                    affine_matrix=matrix,
                    ndim=2,
                )

            # XZ layer: display (X, Z) -> affine translate by (tx, tz)
            if self.projection_xz_b_layer is not None:
                matrix = np.eye(3)
                matrix[0, 2] = tx
                matrix[1, 2] = tz
                self.projection_xz_b_layer.affine = Affine(
                    affine_matrix=matrix,
                    ndim=2,
                )
        finally:
            self._syncing_transforms = False

    def _update_pre_reg_label(self):
        """Update the pre-reg transform label, displaying values in micrometers."""
        tz, ty, tx = self._pre_reg_transform
        self.pre_reg_label.setText(
            f"Pre-reg transform: ({tz:.1f}, {ty:.1f}, {tx:.1f}) μm"
        )

    def _reset_transform(self):
        """Reset the pre-reg transform to [0, 0, 0] and reset View B layer positions."""
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        self._sync_view_b_layers()
        self._update_pre_reg_label()

    def update_progress(self, message):
        """Update progress message."""
        self.status_label.setText(message)

    def set_input_data(self, data_dict):
        """Set input data from previous step (for backward compatibility)."""
        self.input_data = data_dict
        # Automatically update parameters from input data if available
        if data_dict:
            if "step_size" in data_dict:
                self.step_size_spin.setValue(data_dict["step_size"])

            if "pixel_size" in data_dict:
                self.pixel_size_spin.setValue(data_dict["pixel_size"])

            if "theta" in data_dict:
                theta_rad = data_dict["theta"]
                theta_deg = theta_rad * 180 / math.pi
                self.theta_spin.setValue(theta_deg)

    def _compute_common_crops(self, a_shape, b_shape, t0):
        """Compute crop slices for both volumes to their common overlapping region.

        Uses the pre-reg translation to determine the area where View A and
        translated View B overlap, then returns crop slices for each volume.

        Args:
            a_shape: Shape of View A deskewed volume (Z, Y, X).
            b_shape: Shape of View B deskewed volume (Z, Y, X).
            t0: Pre-reg translation in pixel units [tz, ty, tx].

        Returns:
            Tuple of (a_slices, b_slices, cropped_shape) where each slices is
            a tuple of three slice objects for (Z, Y, X) dimensions.
            Returns (None, None, None) if no valid overlap exists.
        """
        Za, Ya, Xa = a_shape
        Zb, Yb, Xb = b_shape
        tz, ty, tx = t0

        # Compute overlap in View A's coordinate system
        # View A occupies [0, Za] x [0, Ya] x [0, Xa]
        # View B translated occupies [tz, tz+Zb] x [ty, ty+Yb] x [tx, tx+Xb]
        z_start_a = max(0, tz)
        z_end_a = min(Za, tz + Zb)
        y_start_a = max(0, ty)
        y_end_a = min(Ya, ty + Yb)
        x_start_a = max(0, tx)
        x_end_a = min(Xa, tx + Xb)

        # Check if valid overlap exists
        if z_start_a >= z_end_a or y_start_a >= y_end_a or x_start_a >= x_end_a:
            return None, None, None

        # Crop slices for View A (directly from overlap in A's coordinates)
        a_slices = (
            slice(int(z_start_a), int(z_end_a)),
            slice(int(y_start_a), int(y_end_a)),
            slice(int(x_start_a), int(x_end_a)),
        )

        # Crop slices for View B (shifted back to B's own coordinates by subtracting translation)
        z_start_b = max(0, -tz)
        z_end_b = min(Zb, Za - tz)
        y_start_b = max(0, -ty)
        y_end_b = min(Yb, Ya - ty)
        x_start_b = max(0, -tx)
        x_end_b = min(Xb, Xa - tx)

        b_slices = (
            slice(int(z_start_b), int(z_end_b)),
            slice(int(y_start_b), int(y_end_b)),
            slice(int(x_start_b), int(x_end_b)),
        )

        cropped_shape = (
            int(z_end_a - z_start_a),
            int(y_end_a - y_start_a),
            int(x_end_a - x_start_a),
        )
        return a_slices, b_slices, cropped_shape

    def register_data(self):
        """Register the deskewed data using background worker."""
        if self.a_deskewed is None or self.b_deskewed is None:
            QMessageBox.warning(
                self, "Error", "Please load and deskew data first"
            )
            return

        transform_type = self.transform_combo.currentText()

        self.register_button.setEnabled(False)
        self.load_deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Starting registration...")

        # Convert pre-reg transform from micrometers to pixel units
        pixel_size = self.pixel_size_spin.value()
        t0 = [t / pixel_size for t in self._pre_reg_transform]

        # Determine volumes to register and initial translation
        a_vol = self.a_deskewed
        b_vol = self.b_deskewed
        initial_t = list(t0)

        # Apply cropping if "Crop to Common" is enabled
        crop_enabled = self.crop_to_common_checkbox.isChecked()
        if crop_enabled:
            a_slices, b_slices, cropped_shape = self._compute_common_crops(
                self.a_deskewed.shape, self.b_deskewed.shape, t0
            )
            if a_slices is None:
                QMessageBox.warning(
                    self,
                    "No Common Region",
                    "The pre-reg translation results in no overlapping region "
                    "between View A and View B. Disabling crop and proceeding "
                    "with full volumes.",
                )
                self.crop_to_common_checkbox.setChecked(False)
                self.update_progress("No common region found - using full volumes")
            else:
                a_vol = self.a_deskewed[a_slices]
                b_vol = self.b_deskewed[b_slices]
                initial_t = [0, 0, 0]  # Pre-reg translation accounted for by crop
                self.update_progress(
                    f"Cropped to common region: "
                    f"{cropped_shape[0]}x{cropped_shape[1]}x{cropped_shape[2]}"
                )

        # Create and start worker
        self.reg_worker = RegistrationWorker(
            a_vol, b_vol, transform_type, initial_translation=initial_t
        )
        self.reg_worker.registered.connect(self.on_registered)
        self.reg_worker.error_occurred.connect(self.on_error)
        self.reg_worker.progress_updated.connect(self.update_progress)
        self.reg_worker.start()

    def on_registered(self, result):
        """Handle successful registration."""
        a_registered = result["a_registered"]
        b_registered = result["b_registered"]
        transform_matrix = result["transform_matrix"]
        cr = result["correlation_ratio"]

        # Add registered data as new layers
        output_a_name = "deskewed_A_registered"
        output_b_name = "deskewed_B_registered"

        # Remove old layers if they exist
        for layer_name in [output_a_name, output_b_name]:
            try:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass

        self.viewer.add_image(
            a_registered,
            name=output_a_name,
            metadata={
                "transform_matrix": transform_matrix,
                "correlation_ratio": cr,
                "transform_type": result["transform_type"],
                "step_size": self.step_size_spin.value(),
                "pixel_size": self.pixel_size_spin.value(),
                "theta": self.theta_spin.value() * math.pi / 180,
            },
        )

        self.viewer.add_image(
            b_registered,
            name=output_b_name,
            metadata={
                "transform_matrix": transform_matrix,
                "correlation_ratio": cr,
                "transform_type": result["transform_type"],
                "step_size": self.step_size_spin.value(),
                "pixel_size": self.pixel_size_spin.value(),
                "theta": self.theta_spin.value() * math.pi / 180,
            },
        )

        # Update status and results
        self.status_label.setText(
            f"Registration completed! A: {a_registered.shape}, B: {b_registered.shape}"
        )

        results_text = f"""
        <b>Registration Results:</b><br>
        Transform Type: {result["transform_type"]}<br>
        Correlation Ratio: {cr:.3f}<br>
        A shape: {a_registered.shape}<br>
        B shape: {b_registered.shape}
        """
        self.results_label.setText(results_text)

        self.register_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with registered data
        output_data = {
            "a_registered": a_registered,
            "b_registered": b_registered,
            "transform_matrix": transform_matrix,
            "correlation_ratio": cr,
            "transform_type": result["transform_type"],
            "step_size": self.step_size_spin.value(),
            "pixel_size": self.pixel_size_spin.value(),
            "theta": self.theta_spin.value() * math.pi / 180,
        }
        self.registered.emit(output_data)

    def on_error(self, error_msg):
        """Handle error."""
        QMessageBox.critical(self, "Error", f"Operation failed: {error_msg}")
        self.status_label.setText("Error during operation")
        self.load_deskew_button.setEnabled(True)
        self.register_button.setEnabled(True)
        self.progress_bar.setVisible(False)
