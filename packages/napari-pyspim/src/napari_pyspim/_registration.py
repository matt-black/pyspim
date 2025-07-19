"""
Registration widget for aligning dual-view SPIM data.
"""

import math
try:
    import cupy as cp
except ImportError:
    import numpy as cp
    print("Warning: CuPy not available, using NumPy (CPU only)")
import numpy as np
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
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
        use_pcc=True,
        upsample_factor=1,
    ):
        super().__init__()
        self.a_deskewed = a_deskewed
        self.b_deskewed = b_deskewed
        self.transform_type = transform_type
        self.use_pcc = use_pcc
        self.upsample_factor = upsample_factor

    def run(self):
        """Perform registration in background thread."""
        try:
            # Lazy imports to avoid CUDA compilation at module level
            from pyspim.interp import affine
            from pyspim.reg import pcc, powell
            from pyspim.util import launch_params_for_volume, pad_to_same_size

            self.progress_updated.emit("Padding volumes to same size...")

            # Pad volumes to same size
            a_dsk, b_dsk = pad_to_same_size(self.a_deskewed, self.b_deskewed)

            # Phase cross correlation for initial guess
            if self.use_pcc:
                self.progress_updated.emit("Computing phase cross correlation...")
                t0 = pcc.translation_for_volumes(
                    a_dsk, b_dsk, upsample_factor=self.upsample_factor
                )
            else:
                t0 = [0, 0, 0]

            # Set up initial parameters and bounds
            self.progress_updated.emit("Setting up optimization parameters...")

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
                "use_pcc": self.use_pcc,
            }

            self.registered.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class RegistrationWidget(QWidget):
    """Widget for registering dual-view data."""

    registered = pyqtSignal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Layer selection
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QFormLayout()

        self.layer_a_combo = QComboBox()
        self.layer_a_combo.addItem("Select Channel A layer...")
        self.layer_a_combo.currentTextChanged.connect(self.on_layer_selection_changed)

        self.layer_b_combo = QComboBox()
        self.layer_b_combo.addItem("Select Channel B layer...")
        self.layer_b_combo.currentTextChanged.connect(self.on_layer_selection_changed)

        layer_layout.addRow("Channel A:", self.layer_a_combo)
        layer_layout.addRow("Channel B:", self.layer_b_combo)
        layer_group.setLayout(layer_layout)

        # Registration parameters
        params_group = QGroupBox("Registration Parameters")
        params_layout = QFormLayout()

        self.transform_combo = QComboBox()
        self.transform_combo.addItems(["t", "t+r", "t+r+s"])
        self.transform_combo.setCurrentText("t+r+s")

        self.use_pcc_check = QComboBox()
        self.use_pcc_check.addItems(["Yes", "No"])
        self.use_pcc_check.setCurrentText("Yes")

        self.upsample_spin = QSpinBox()
        self.upsample_spin.setRange(1, 10)
        self.upsample_spin.setValue(1)

        params_layout.addRow("Transform Type:", self.transform_combo)
        params_layout.addRow("Use Phase Correlation:", self.use_pcc_check)
        params_layout.addRow("Upsample Factor:", self.upsample_spin)
        params_group.setLayout(params_layout)

        # Acquisition parameters (for reference and metadata)
        acq_group = QGroupBox("Acquisition Parameters")
        acq_layout = QFormLayout()

        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.1, 10.0)
        self.step_size_spin.setValue(0.5)
        self.step_size_spin.setSuffix(" μm")
        self.step_size_spin.setDecimals(3)

        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.01, 1.0)
        self.pixel_size_spin.setValue(0.1625)
        self.pixel_size_spin.setSuffix(" μm")
        self.pixel_size_spin.setDecimals(4)

        self.theta_spin = QDoubleSpinBox()
        self.theta_spin.setRange(0, 90)
        self.theta_spin.setValue(45)
        self.theta_spin.setSuffix("°")
        self.theta_spin.setDecimals(1)

        acq_layout.addRow("Step Size:", self.step_size_spin)
        acq_layout.addRow("Pixel Size:", self.pixel_size_spin)
        acq_layout.addRow("Theta (angle):", self.theta_spin)
        acq_group.setLayout(acq_layout)

        # Register button
        self.register_button = QPushButton("Register Data")
        self.register_button.clicked.connect(self.register_data)
        self.register_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("Select layers to register")

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(layer_group)
        layout.addWidget(params_group)
        layout.addWidget(acq_group)
        layout.addWidget(self.register_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_label)
        layout.addStretch()

        self.setLayout(layout)

        # Update layer lists when viewer layers change
        self.viewer.layers.events.inserted.connect(self.update_layer_lists)
        self.viewer.layers.events.removed.connect(self.update_layer_lists)

    def update_layer_lists(self, event=None):
        """Update the layer selection dropdowns."""
        # Store current selections
        current_a = self.layer_a_combo.currentText()
        current_b = self.layer_b_combo.currentText()

        # Clear and repopulate
        self.layer_a_combo.clear()
        self.layer_b_combo.clear()

        # Add placeholder items
        self.layer_a_combo.addItem("Select Channel A layer...")
        self.layer_b_combo.addItem("Select Channel B layer...")

        # Add available layers
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and layer.data is not None:
                self.layer_a_combo.addItem(layer.name)
                self.layer_b_combo.addItem(layer.name)

        # Restore selections if they still exist
        if current_a and current_a in [self.layer_a_combo.itemText(i) for i in range(self.layer_a_combo.count())]:
            self.layer_a_combo.setCurrentText(current_a)
        if current_b and current_b in [self.layer_b_combo.itemText(i) for i in range(self.layer_b_combo.count())]:
            self.layer_b_combo.setCurrentText(current_b)

        self.on_layer_selection_changed()

    def on_layer_selection_changed(self):
        """Handle layer selection changes."""
        a_selected = self.layer_a_combo.currentText() != "Select Channel A layer..."
        b_selected = self.layer_b_combo.currentText() != "Select Channel B layer..."

        # Enable register button if both layers are selected
        self.register_button.setEnabled(a_selected and b_selected)

        # Update parameters from selected layers
        self._update_parameters_from_selected_layers()

        # Update status
        if a_selected and b_selected:
            self.status_label.setText("Ready to register both channels")
        elif a_selected:
            self.status_label.setText("Please select Channel B layer")
        elif b_selected:
            self.status_label.setText("Please select Channel A layer")
        else:
            self.status_label.setText("Select layers to register")

    def _update_parameters_from_selected_layers(self):
        """Update parameters from selected layer metadata."""
        # Check both selected layers for metadata
        layers_to_check = []
        
        if self.layer_a_combo.currentText() != "Select Channel A layer...":
            try:
                layer_a = self.viewer.layers[self.layer_a_combo.currentText()]
                layers_to_check.append(layer_a)
            except (KeyError, ValueError):
                pass

        if self.layer_b_combo.currentText() != "Select Channel B layer...":
            try:
                layer_b = self.viewer.layers[self.layer_b_combo.currentText()]
                layers_to_check.append(layer_b)
            except (KeyError, ValueError):
                pass

        # Try to get parameters from layer metadata
        for layer in layers_to_check:
            if hasattr(layer, 'metadata') and layer.metadata:
                metadata = layer.metadata
                
                if 'step_size' in metadata:
                    self.step_size_spin.setValue(metadata['step_size'])
                
                if 'pixel_size' in metadata:
                    self.pixel_size_spin.setValue(metadata['pixel_size'])
                
                if 'theta' in metadata:
                    theta_rad = metadata['theta']
                    theta_deg = theta_rad * 180 / math.pi
                    self.theta_spin.setValue(theta_deg)
                
                # Found parameters, no need to check other layers
                break

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

    def register_data(self):
        """Register the data using background worker."""
        # Get selected layers
        a_layer_name = self.layer_a_combo.currentText()
        b_layer_name = self.layer_b_combo.currentText()
        
        if a_layer_name == "Select Channel A layer..." or b_layer_name == "Select Channel B layer...":
            QMessageBox.warning(self, "Error", "Please select both layers to register")
            return

        # Get data from selected layers
        try:
            a_layer = self.viewer.layers[a_layer_name]
            b_layer = self.viewer.layers[b_layer_name]
            a_deskewed = a_layer.data
            b_deskewed = b_layer.data
        except (KeyError, ValueError) as e:
            QMessageBox.warning(self, "Error", f"Could not get data from selected layers: {e}")
            return

        transform_type = self.transform_combo.currentText()
        use_pcc = self.use_pcc_check.currentText() == "Yes"
        upsample_factor = self.upsample_spin.value()

        self.register_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Starting registration...")

        # Create and start worker
        self.worker = RegistrationWorker(
            a_deskewed, b_deskewed, transform_type, use_pcc, upsample_factor
        )
        self.worker.registered.connect(self.on_registered)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, message):
        """Update progress message."""
        self.status_label.setText(message)

    def on_registered(self, result):
        """Handle successful registration."""
        a_registered = result["a_registered"]
        b_registered = result["b_registered"]
        transform_matrix = result["transform_matrix"]
        cr = result["correlation_ratio"]

        # Add registered data as new layers
        source_a_name = self.layer_a_combo.currentText()
        source_b_name = self.layer_b_combo.currentText()
        
        output_a_name = f"{source_a_name}_registered"
        output_b_name = f"{source_b_name}_registered"
        
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
        Use Phase Correlation: {result["use_pcc"]}<br>
        A shape: {a_registered.shape}<br>
        B shape: {b_registered.shape}
        """
        self.results_label.setText(results_text)

        self.register_button.setEnabled(True)
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
        """Handle registration error."""
        QMessageBox.critical(self, "Error", f"Registration failed: {error_msg}")
        self.status_label.setText("Error during registration")
        self.register_button.setEnabled(True)
        self.progress_bar.setVisible(False)
