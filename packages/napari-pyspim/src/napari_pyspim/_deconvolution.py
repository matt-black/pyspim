"""
Deconvolution widget for Richardson-Lucy dual-view deconvolution.
"""

import os

import cupy as cp
import numpy as np
import zarr
from PyQt5.QtCore import pyqtSignal
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
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

# Lazy import to avoid CUDA compilation at module level
# from pyspim.decon.rl.dualview_fft import deconvolve_chunkwise


class DeconvolutionWorker(QThread):
    """Worker thread for deconvolution."""

    deconvolved = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str)

    def __init__(
        self,
        a_registered,
        b_registered,
        psf_a,
        psf_b,
        chunk_size,
        overlap,
        iterations,
        regularization,
        noise_model="additive",
        output_path=None,
    ):
        super().__init__()
        self.a_registered = a_registered
        self.b_registered = b_registered
        self.psf_a = psf_a
        self.psf_b = psf_b
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.iterations = iterations
        self.regularization = regularization
        self.noise_model = noise_model
        self.output_path = output_path

    def run(self):
        """Perform deconvolution in background thread."""
        try:
            # Lazy import to avoid CUDA compilation at module level
            from pyspim.decon.rl.dualview_fft import deconvolve_chunkwise

            self.progress_updated.emit("Setting up deconvolution...")

            # Create temporary zarr arrays
            temp_dir = "/tmp/pyspim_decon" if not self.output_path else self.output_path
            os.makedirs(temp_dir, exist_ok=True)

            a_zarr_path = os.path.join(temp_dir, "a.zarr")
            b_zarr_path = os.path.join(temp_dir, "b.zarr")
            out_zarr_path = os.path.join(temp_dir, "out.zarr")

            # Save input data to zarr
            self.progress_updated.emit("Preparing input data...")
            a_zarr = zarr.creation.open_array(
                a_zarr_path,
                mode="w",
                shape=self.a_registered.shape,
                dtype=np.uint16,
                fill_value=0,
            )
            a_zarr[:] = self.a_registered

            b_zarr = zarr.creation.open_array(
                b_zarr_path,
                mode="w",
                shape=self.b_registered.shape,
                dtype=np.uint16,
                fill_value=0,
            )
            b_zarr[:] = self.b_registered

            # Create output zarr
            out_zarr = zarr.creation.open_array(
                out_zarr_path,
                mode="w",
                shape=self.b_registered.shape,
                dtype=np.float32,
                fill_value=0,
            )

            # Prepare PSFs
            psf_a_gpu = cp.asarray(self.psf_a)
            psf_b_gpu = cp.asarray(self.psf_b)
            psf_a_flipped = cp.asarray(self.psf_a[::-1, ::-1, ::-1])
            psf_b_flipped = cp.asarray(self.psf_b[::-1, ::-1, ::-1])

            # Perform chunked deconvolution
            self.progress_updated.emit("Starting deconvolution...")
            deconvolve_chunkwise(
                a_zarr,
                b_zarr,
                out_zarr,
                self.chunk_size,
                self.overlap,
                psf_a_gpu,
                psf_b_gpu,
                psf_a_flipped,
                psf_b_flipped,
                self.noise_model,
                self.iterations,
                self.regularization,
                False,
                None,
                0,
                0,
                True,
            )

            # Load result
            self.progress_updated.emit("Loading deconvolved result...")
            decon_result = zarr.load(out_zarr_path)

            # Clean up temporary files
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

            result = {
                "deconvolved": decon_result,
                "iterations": self.iterations,
                "regularization": self.regularization,
                "noise_model": self.noise_model,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
            }

            self.deconvolved.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class DeconvolutionWidget(QWidget):
    """Widget for Richardson-Lucy deconvolution."""

    deconvolved = Signal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.worker = None
        self.input_data = None
        self.psf_a = None
        self.psf_b = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # PSF loading
        psf_group = QGroupBox("Point Spread Functions")
        psf_layout = QFormLayout()

        self.psf_a_path = QLineEdit()
        self.psf_a_path.setPlaceholderText("Select PSF A file (.npy)")
        self.psf_a_browse = QPushButton("Browse")
        self.psf_a_browse.clicked.connect(lambda: self.browse_psf("a"))

        self.psf_b_path = QLineEdit()
        self.psf_b_path.setPlaceholderText("Select PSF B file (.npy)")
        self.psf_b_browse = QPushButton("Browse")
        self.psf_b_browse.clicked.connect(lambda: self.browse_psf("b"))

        psf_a_row = QHBoxLayout()
        psf_a_row.addWidget(self.psf_a_path)
        psf_a_row.addWidget(self.psf_a_browse)

        psf_b_row = QHBoxLayout()
        psf_b_row.addWidget(self.psf_b_path)
        psf_b_row.addWidget(self.psf_b_browse)

        psf_layout.addRow("PSF A:", psf_a_row)
        psf_layout.addRow("PSF B:", psf_b_row)
        psf_group.setLayout(psf_layout)

        # Deconvolution parameters
        params_group = QGroupBox("Deconvolution Parameters")
        params_layout = QFormLayout()

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(20)

        self.regularization_spin = QDoubleSpinBox()
        self.regularization_spin.setRange(1e-10, 1e-3)
        self.regularization_spin.setValue(1e-6)
        self.regularization_spin.setDecimals(10)
        self.regularization_spin.setSingleStep(1e-6)

        self.noise_model_combo = QComboBox()
        self.noise_model_combo.addItems(["additive", "poisson"])
        self.noise_model_combo.setCurrentText("additive")

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(64, 1024)
        self.chunk_size_spin.setValue(128)
        self.chunk_size_spin.setSuffix(" pixels")

        self.overlap_spin = QSpinBox()
        self.overlap_spin.setRange(10, 100)
        self.overlap_spin.setValue(40)
        self.overlap_spin.setSuffix(" pixels")

        params_layout.addRow("Iterations:", self.iterations_spin)
        params_layout.addRow("Regularization:", self.regularization_spin)
        params_layout.addRow("Noise Model:", self.noise_model_combo)
        params_layout.addRow("Chunk Size:", self.chunk_size_spin)
        params_layout.addRow("Overlap:", self.overlap_spin)
        params_group.setLayout(params_layout)

        # Deconvolve button
        self.deconvolve_button = QPushButton("Deconvolve Data")
        self.deconvolve_button.clicked.connect(self.deconvolve_data)
        self.deconvolve_button.setEnabled(False)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Status label
        self.status_label = QLabel("No input data available")

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(psf_group)
        layout.addWidget(params_group)
        layout.addWidget(self.deconvolve_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.results_label)
        layout.addStretch()

        self.setLayout(layout)

    def browse_psf(self, psf_type):
        """Browse for PSF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select PSF {psf_type.upper()} file", "", "NumPy files (*.npy)"
        )
        if file_path:
            if psf_type == "a":
                self.psf_a_path.setText(file_path)
                self.psf_a = np.load(file_path)
            else:
                self.psf_b_path.setText(file_path)
                self.psf_b = np.load(file_path)
            self._check_ready()

    def set_input_data(self, data_dict):
        """Set input data from previous step."""
        self.input_data = data_dict
        self._check_ready()
        self.status_label.setText("Input data ready for deconvolution")

    def _check_ready(self):
        """Check if all inputs are ready for deconvolution."""
        ready = (
            self.input_data is not None
            and self.psf_a is not None
            and self.psf_b is not None
        )
        self.deconvolve_button.setEnabled(ready)

        if ready:
            self.status_label.setText("Ready to deconvolve")
        else:
            missing = []
            if not self.input_data:
                missing.append("registered data")
            if self.psf_a is None:
                missing.append("PSF A")
            if self.psf_b is None:
                missing.append("PSF B")
            self.status_label.setText(f"Missing: {', '.join(missing)}")

    def deconvolve_data(self):
        """Deconvolve the data using background worker."""
        if not (
            self.input_data is not None
            and self.psf_a is not None
            and self.psf_b is not None
        ):
            QMessageBox.warning(self, "Error", "Please provide all required inputs")
            return

        a_registered = self.input_data["a_registered"]
        b_registered = self.input_data["b_registered"]

        iterations = self.iterations_spin.value()
        regularization = self.regularization_spin.value()
        noise_model = self.noise_model_combo.currentText()
        chunk_size = [self.chunk_size_spin.value()] * 3
        overlap = [self.overlap_spin.value()] * 3

        self.deconvolve_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Starting deconvolution...")

        # Create and start worker
        self.worker = DeconvolutionWorker(
            a_registered,
            b_registered,
            self.psf_a,
            self.psf_b,
            chunk_size,
            overlap,
            iterations,
            regularization,
            noise_model,
        )
        self.worker.deconvolved.connect(self.on_deconvolved)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.start()

    def update_progress(self, message):
        """Update progress message."""
        self.status_label.setText(message)

    def on_deconvolved(self, result):
        """Handle successful deconvolution."""
        deconvolved = result["deconvolved"]

        # Remove old layer if it exists
        try:
            layer = self.viewer.layers["Deconvolved"]
            self.viewer.layers.remove(layer)
        except (KeyError, ValueError):
            pass

        # Add deconvolved data as new layer
        self.viewer.add_image(
            deconvolved,
            name="Deconvolved",
            metadata={
                "iterations": result["iterations"],
                "regularization": result["regularization"],
                "noise_model": result["noise_model"],
                "chunk_size": result["chunk_size"],
                "overlap": result["overlap"],
            },
        )

        # Update status and results
        self.status_label.setText(
            f"Deconvolution completed! Shape: {deconvolved.shape}"
        )

        results_text = f"""
        <b>Deconvolution Results:</b><br>
        Iterations: {result["iterations"]}<br>
        Regularization: {result["regularization"]:.2e}<br>
        Noise Model: {result["noise_model"]}<br>
        Chunk Size: {result["chunk_size"]}<br>
        Overlap: {result["overlap"]}<br>
        Output Shape: {deconvolved.shape}<br>
        Memory Usage: {deconvolved.size * 4 / 1e9:.2f} GB
        """
        self.results_label.setText(results_text)

        self.deconvolve_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with deconvolved data
        self.deconvolved.emit(result)

    def on_error(self, error_msg):
        """Handle deconvolution error."""
        QMessageBox.critical(self, "Error", f"Deconvolution failed: {error_msg}")
        self.status_label.setText("Error during deconvolution")
        self.deconvolve_button.setEnabled(True)
        self.progress_bar.setVisible(False)
