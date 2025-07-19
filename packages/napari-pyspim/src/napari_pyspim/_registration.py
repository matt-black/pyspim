"""
Registration widget for aligning dual-view SPIM data.
"""

import cupy as cp
import numpy as np
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
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

    registered = Signal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

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

        # Register button
        self.register_button = QPushButton("Register Data")
        self.register_button.clicked.connect(self.register_data)
        self.register_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("No input data available")

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(params_group)
        layout.addWidget(self.register_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_label)
        layout.addStretch()

        self.setLayout(layout)

    def set_input_data(self, data_dict):
        """Set input data from previous step."""
        self.input_data = data_dict
        self.register_button.setEnabled(True)
        self.status_label.setText("Input data ready for registration")

    def register_data(self):
        """Register the data using background worker."""
        if not self.input_data:
            QMessageBox.warning(self, "Error", "No input data available")
            return

        a_deskewed = self.input_data["a_deskewed"]
        b_deskewed = self.input_data["b_deskewed"]
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

        # Remove old layers if they exist
        for layer_name in ["A_registered", "B_registered"]:
            try:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass

        # Add registered data as new layers
        self.viewer.add_image(
            a_registered,
            name="A_registered",
            metadata={
                "transform_matrix": transform_matrix,
                "correlation_ratio": cr,
                "transform_type": result["transform_type"],
            },
        )

        self.viewer.add_image(
            b_registered,
            name="B_registered",
            metadata={
                "transform_matrix": transform_matrix,
                "correlation_ratio": cr,
                "transform_type": result["transform_type"],
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
        }
        self.registered.emit(output_data)

    def on_error(self, error_msg):
        """Handle registration error."""
        QMessageBox.critical(self, "Error", f"Registration failed: {error_msg}")
        self.status_label.setText("Error during registration")
        self.register_button.setEnabled(True)
        self.progress_bar.setVisible(False)
