"""
Deconvolution widget for Richardson-Lucy dual-view deconvolution.
"""

import math
import os
import tempfile
from typing import Optional, Tuple

import numpy
import tifffile
import zarr
from scipy import ndimage

from qtpy.QtCore import Signal

from ._psf import generate_psf_im, normalize_psf_im
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
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def _fwhm_to_sigma(fwhm: float) -> float:
    """Convert full-width at half-maximum to standard deviation for a Gaussian."""
    return fwhm / (2 * numpy.sqrt(2 * numpy.log(2)))


def make_gaussian_psf_volume(
    fwhm_ax: float,
    fwhm_lat: float,
    size_ax: int,
    size_lat: int,
    theta: float = 0.0,
) -> numpy.ndarray:
    """Generate a Gaussian PSF volume for deconvolution.

    Args:
        fwhm_ax: Full-width at half-maximum in the axial direction.
        fwhm_lat: Full-width at half-maximum in the lateral direction.
        size_ax: Size of the volume in the axial dimension.
        size_lat: Size of the volume in each lateral dimension.
        theta: Rotation angle in degrees in the ZX plane. theta=0 means
            no rotation (PSF pointing up when fwhm_ax > fwhm_lat).

    Returns:
        A numpy.ndarray of shape (size_ax, size_lat, size_lat) containing
        the Gaussian PSF volume, centered and normalized to unit total intensity.
    """
    # Convert FWHM to sigma
    sigma_ax = _fwhm_to_sigma(fwhm_ax)
    sigma_lat = _fwhm_to_sigma(fwhm_lat)

    # Generate PSF volume centered in the volume
    # pars: [x0, y0, z0, sx, sy, sz, ampl, bkgrnd]
    im_shape = (size_ax, size_lat, size_lat)
    pars = [size_lat / 2, size_lat / 2, size_ax / 2, sigma_lat, sigma_lat, sigma_ax, 1.0, 0.0]
    psf_volume = generate_psf_im(pars, im_shape, "spherical")

    # Apply rotation in the ZX plane (axes 0 and 2) if theta is non-zero
    if theta != 0.0:
        psf_volume = ndimage.rotate(
            psf_volume, theta, axes=(0, 2), reshape=True,
            mode="constant", cval=0
        )

    # Normalize to unit total intensity
    psf_volume = normalize_psf_im(psf_volume)

    return psf_volume


class DeconvolutionWorker(QThread):
    """Worker thread for deconvolution."""

    finished = Signal(dict)
    error_occurred = Signal(str)
    progress_updated = Signal(str, int)  # message, percentage (0-100)

    def __init__(
        self,
        view_a: numpy.ndarray,
        view_b: numpy.ndarray,
        psf_a: numpy.ndarray,
        psf_b: numpy.ndarray,
        backproj_a: numpy.ndarray,
        backproj_b: numpy.ndarray,
        decon_function: str,
        num_iter: int,
        epsilon: float,
        req_both: bool,
        boundary_correction: bool,
        boundary_sigma: float,
        chunkwise: bool,
        chunk_size: Tuple[int, int, int],
        overlap: Tuple[int, int, int],
        save_path: Optional[str],
        use_remote: bool = False,
        remote_client = None,
    ):
        super().__init__()
        self.view_a = view_a
        self.view_b = view_b
        self.psf_a = psf_a
        self.psf_b = psf_b
        self.backproj_a = backproj_a
        self.backproj_b = backproj_b
        self.decon_function = decon_function
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.req_both = req_both
        self.boundary_correction = boundary_correction
        self.boundary_sigma = boundary_sigma
        self.chunkwise = chunkwise
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.save_path = save_path
        self.use_remote = use_remote
        self.remote_client = remote_client

    def run(self):
        """Perform deconvolution in background thread."""
        if self.use_remote:
            self._run_remote()
        else:
            self._run_local()

    def _run_local(self):
        """Perform deconvolution locally."""
        try:
            # Lazy imports to avoid CUDA compilation at module level
            import cupy
            from pyspim.decon.rl.dualview_fft import deconvolve, deconvolve_chunkwise

            if self.chunkwise:
                self.progress_updated.emit("Starting chunkwise deconvolution...", 0)
                result = self._run_chunkwise()
            else:
                self.progress_updated.emit("Starting deconvolution...", 0)
                result = self._run_full()

            # Save output if requested
            if self.save_path is not None:
                self.progress_updated.emit("Saving output...", 90)
                self._save_output(result)

            self.progress_updated.emit("Deconvolution completed!", 100)
            self.finished.emit({"result": result, "save_path": self.save_path})

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _run_full(self) -> numpy.ndarray:
        """Run full (non-chunkwise) deconvolution."""
        import cupy
        from pyspim.decon.rl.dualview_fft import deconvolve

        self.progress_updated.emit("Converting data to GPU...", 5)

        # Convert to cupy float32 arrays
        view_a = cupy.asarray(self.view_a, dtype=cupy.float32)
        view_b = cupy.asarray(self.view_b, dtype=cupy.float32)
        psf_a = cupy.asarray(self.psf_a, dtype=cupy.float32)
        psf_b = cupy.asarray(self.psf_b, dtype=cupy.float32)
        backproj_a = cupy.asarray(self.backproj_a, dtype=cupy.float32)
        backproj_b = cupy.asarray(self.backproj_b, dtype=cupy.float32)

        self.progress_updated.emit("Running deconvolution iterations...", 10)

        result = deconvolve(
            view_a=view_a,
            view_b=view_b,
            est_i=None,
            psf_a=psf_a,
            psf_b=psf_b,
            backproj_a=backproj_a,
            backproj_b=backproj_b,
            decon_function=self.decon_function,
            num_iter=self.num_iter,
            epsilon=self.epsilon,
            req_both=self.req_both,
            boundary_correction=self.boundary_correction,
            zero_padding=None,
            boundary_sigma_a=self.boundary_sigma,
            boundary_sigma_b=self.boundary_sigma,
            verbose=False,
        )

        self.progress_updated.emit("Transferring result back...", 80)
        return result.get().astype(numpy.float32)

    def _run_chunkwise(self) -> numpy.ndarray:
        """Run chunkwise deconvolution using temporary zarr arrays."""
        from pyspim.decon.rl.dualview_fft import deconvolve_chunkwise

        self.progress_updated.emit("Preparing chunkwise deconvolution...", 5)

        with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_dir:
            # Create zarr arrays for input and output
            shape = self.view_a.shape
            dtype = numpy.float32

            tmp_a = os.path.join(tmp_dir, "view_a.zarr")
            tmp_b = os.path.join(tmp_dir, "view_b.zarr")
            tmp_out = os.path.join(tmp_dir, "output.zarr")

            zarr_a = zarr.open(tmp_a, mode="w", shape=shape, dtype=dtype)
            zarr_b = zarr.open(tmp_b, mode="w", shape=shape, dtype=dtype)
            zarr_out = zarr.open(tmp_out, mode="w", shape=shape, dtype=dtype, fill_value=0)

            zarr_a[:] = self.view_a
            zarr_b[:] = self.view_b

            self.progress_updated.emit("Running chunkwise deconvolution...", 10)

            # Convert PSF/backprojector arrays to numpy float32 for chunkwise function
            psf_a = numpy.asarray(self.psf_a, dtype=numpy.float32)
            psf_b = numpy.asarray(self.psf_b, dtype=numpy.float32)
            bp_a = numpy.asarray(self.backproj_a, dtype=numpy.float32)
            bp_b = numpy.asarray(self.backproj_b, dtype=numpy.float32)

            deconvolve_chunkwise(
                view_a=zarr_a,
                view_b=zarr_b,
                out=zarr_out,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
                psf_a=psf_a,
                psf_b=psf_b,
                bp_a=bp_a,
                bp_b=bp_b,
                decon_function=self.decon_function,
                num_iter=self.num_iter,
                epsilon=self.epsilon,
                boundary_correction=self.boundary_correction,
                zero_padding=None,
                boundary_sigma_a=self.boundary_sigma,
                boundary_sigma_b=self.boundary_sigma,
                verbose=True,
            )

            self.progress_updated.emit("Reading chunkwise results...", 80)
            return numpy.asarray(zarr_out[:]).astype(numpy.float32)

    def _run_remote(self):
        """Perform deconvolution via remote server."""
        try:
            self.progress_updated.emit("Starting remote deconvolution...", 0)

            result = self.remote_client.send_command_blocking("deconvolve", {
                "view_a": self.view_a,
                "view_b": self.view_b,
                "psf_a": self.psf_a,
                "psf_b": self.psf_b,
                "backproj_a": self.backproj_a,
                "backproj_b": self.backproj_b,
                "decon_function": self.decon_function,
                "num_iter": self.num_iter,
                "epsilon": self.epsilon,
                "req_both": self.req_both,
                "boundary_correction": self.boundary_correction,
                "boundary_sigma": self.boundary_sigma,
                "chunkwise": self.chunkwise,
                "chunk_size": list(self.chunk_size),
                "overlap": list(self.overlap),
            })

            # Convert shape lists to tuples for compatibility
            if "shape" in result and isinstance(result["shape"], list):
                result["shape"] = tuple(result["shape"])

            self.progress_updated.emit("Remote deconvolution completed!", 100)
            self.finished.emit({"result": result, "save_path": self.save_path})

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _save_output(self, result: numpy.ndarray):
        """Save deconvolution result to a TIFF file."""
        # save_path should not be None here (checked in run()), but be safe
        if self.save_path is None:
            return
        # If save_path is a directory, create a filename
        if os.path.isdir(self.save_path):
            output_file = os.path.join(self.save_path, "deconvolved.ome.tif")
        else:
            output_file = self.save_path

        tifffile.imwrite(
            output_file,
            result,
            bigtiff=True,
            dtype="float32",
            photometric="minisblack",
        )


class DeconvolutionWidget(QWidget):
    """Widget for Richardson-Lucy deconvolution."""

    deconvolved = Signal(dict)

    def __init__(self, viewer, remote_client=None, has_pyspim=True):
        super().__init__()
        self.viewer = viewer
        self.remote_client = remote_client
        self.has_pyspim = has_pyspim
        self.decon_worker = None
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # === Input Data Section ===
        input_group = QGroupBox("Input Data")
        input_layout = QVBoxLayout()

        # Mode toggle: Layers vs Paths
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Input Mode:"))
        self.input_mode_layers = QRadioButton("Layers")
        self.input_mode_layers.setChecked(True)
        self.input_mode_layers.toggled.connect(self.on_input_mode_changed)
        self.input_mode_paths = QRadioButton("Paths")
        self.input_mode_paths.toggled.connect(self.on_input_mode_changed)
        mode_layout.addWidget(self.input_mode_layers)
        mode_layout.addWidget(self.input_mode_paths)
        mode_layout.addStretch()
        input_layout.addLayout(mode_layout)

        # Layers mode widgets
        self.layers_widget = QWidget()
        layers_layout = QFormLayout()

        self.layer_a_combo = QComboBox()
        self.layer_a_combo.addItem("Select View A layer...")

        self.layer_b_combo = QComboBox()
        self.layer_b_combo.addItem("Select View B layer...")

        layers_layout.addRow("View A:", self.layer_a_combo)
        layers_layout.addRow("View B:", self.layer_b_combo)
        self.layers_widget.setLayout(layers_layout)
        input_layout.addWidget(self.layers_widget)

        # Paths mode widgets (hidden by default)
        self.paths_widget = QWidget()
        paths_layout = QFormLayout()

        self.path_a_edit = QLineEdit()
        self.path_a_edit.setPlaceholderText("Select zarr path for View A...")
        self.path_a_browse = QPushButton("Browse")
        self.path_a_browse.clicked.connect(lambda: self.browse_path("a"))

        self.path_b_edit = QLineEdit()
        self.path_b_edit.setPlaceholderText("Select zarr path for View B...")
        self.path_b_browse = QPushButton("Browse")
        self.path_b_browse.clicked.connect(lambda: self.browse_path("b"))

        path_a_row = QHBoxLayout()
        path_a_row.addWidget(self.path_a_edit)
        path_a_row.addWidget(self.path_a_browse)

        path_b_row = QHBoxLayout()
        path_b_row.addWidget(self.path_b_edit)
        path_b_row.addWidget(self.path_b_browse)

        paths_layout.addRow("View A:", path_a_row)
        paths_layout.addRow("View B:", path_b_row)
        self.paths_widget.setLayout(paths_layout)
        self.paths_widget.setVisible(False)
        input_layout.addWidget(self.paths_widget)

        # Deskew Method dropdown
        deskew_layout = QFormLayout()
        self.deskew_method_combo = QComboBox()
        self.deskew_method_combo.addItems(["Orthogonal", "Shear-Warp"])
        deskew_layout.addRow("Deskew Method:", self.deskew_method_combo)
        input_layout.addLayout(deskew_layout)

        input_group.setLayout(input_layout)

        # === Point Spread Function Section ===
        psf_group = QGroupBox("Point Spread Function")
        psf_layout = QVBoxLayout()

        # Type dropdown
        type_layout = QFormLayout()
        self.psf_type_combo = QComboBox()
        self.psf_type_combo.addItems(["Gaussian", "Custom"])
        self.psf_type_combo.currentTextChanged.connect(self.on_psf_type_changed)
        type_layout.addRow("Type:", self.psf_type_combo)
        psf_layout.addLayout(type_layout)

        # Gaussian mode widgets
        self.gaussian_widget = QWidget()
        gaussian_layout = QFormLayout()

        self.psf_a_fwhm_lateral = QDoubleSpinBox()
        self.psf_a_fwhm_lateral.setRange(0.1, 100.0)
        self.psf_a_fwhm_lateral.setDecimals(3)
        self.psf_a_fwhm_lateral.setValue(2.277)
        self.psf_a_fwhm_lateral.setSuffix(" pix")

        self.psf_a_fwhm_axial = QDoubleSpinBox()
        self.psf_a_fwhm_axial.setRange(0.1, 500.0)
        self.psf_a_fwhm_axial.setDecimals(3)
        self.psf_a_fwhm_axial.setValue(7.385)
        self.psf_a_fwhm_axial.setSuffix(" pix")

        self.psf_b_fwhm_lateral = QDoubleSpinBox()
        self.psf_b_fwhm_lateral.setRange(0.1, 100.0)
        self.psf_b_fwhm_lateral.setDecimals(3)
        self.psf_b_fwhm_lateral.setValue(2.277)
        self.psf_b_fwhm_lateral.setSuffix(" pix")

        self.psf_b_fwhm_axial = QDoubleSpinBox()
        self.psf_b_fwhm_axial.setRange(0.1, 500.0)
        self.psf_b_fwhm_axial.setDecimals(3)
        self.psf_b_fwhm_axial.setValue(7.385)
        self.psf_b_fwhm_axial.setSuffix(" pix")

        gaussian_layout.addRow("A: FWHM, Lateral:", self.psf_a_fwhm_lateral)
        gaussian_layout.addRow("A: FWHM, Axial:", self.psf_a_fwhm_axial)
        gaussian_layout.addRow("B: FWHM, Lateral:", self.psf_b_fwhm_lateral)
        gaussian_layout.addRow("B: FWHM, Axial:", self.psf_b_fwhm_axial)
        self.gaussian_widget.setLayout(gaussian_layout)
        psf_layout.addWidget(self.gaussian_widget)

        # Custom mode widgets (hidden by default)
        self.custom_psf_widget = QWidget()
        custom_psf_layout = QFormLayout()

        self.psf_a_path = QLineEdit()
        self.psf_a_path.setPlaceholderText("Select PSF A file (.npy, .tif)")
        self.psf_a_browse = QPushButton("Browse")
        self.psf_a_browse.clicked.connect(lambda: self.browse_psf("a"))

        self.psf_b_path = QLineEdit()
        self.psf_b_path.setPlaceholderText("Select PSF B file (.npy, .tif)")
        self.psf_b_browse = QPushButton("Browse")
        self.psf_b_browse.clicked.connect(lambda: self.browse_psf("b"))

        psf_a_row = QHBoxLayout()
        psf_a_row.addWidget(self.psf_a_path)
        psf_a_row.addWidget(self.psf_a_browse)

        psf_b_row = QHBoxLayout()
        psf_b_row.addWidget(self.psf_b_path)
        psf_b_row.addWidget(self.psf_b_browse)

        custom_psf_layout.addRow("PSF A:", psf_a_row)
        custom_psf_layout.addRow("PSF B:", psf_b_row)
        self.custom_psf_widget.setLayout(custom_psf_layout)
        self.custom_psf_widget.setVisible(False)
        psf_layout.addWidget(self.custom_psf_widget)

        # "Backprojector is Flipped PSF" checkbox
        self.flipped_psf_check = QCheckBox("Backprojector is Flipped PSF")
        self.flipped_psf_check.setChecked(True)
        self.flipped_psf_check.toggled.connect(self.on_flipped_psf_toggled)
        psf_layout.addWidget(self.flipped_psf_check)

        # Backprojectors subsection (hidden when "Backprojector is Flipped PSF" is checked)
        self.backprojector_group = QGroupBox("Backprojectors")
        self.backprojector_layout = QVBoxLayout()

        # Backprojector Gaussian widgets
        self.backprojector_gaussian_widget = QWidget()
        bp_gaussian_layout = QFormLayout()

        self.bp_a_fwhm_lateral = QDoubleSpinBox()
        self.bp_a_fwhm_lateral.setRange(0.1, 100.0)
        self.bp_a_fwhm_lateral.setDecimals(3)
        self.bp_a_fwhm_lateral.setValue(2.277)
        self.bp_a_fwhm_lateral.setSuffix(" pix")

        self.bp_a_fwhm_axial = QDoubleSpinBox()
        self.bp_a_fwhm_axial.setRange(0.1, 500.0)
        self.bp_a_fwhm_axial.setDecimals(3)
        self.bp_a_fwhm_axial.setValue(7.385)
        self.bp_a_fwhm_axial.setSuffix(" pix")

        self.bp_b_fwhm_lateral = QDoubleSpinBox()
        self.bp_b_fwhm_lateral.setRange(0.1, 100.0)
        self.bp_b_fwhm_lateral.setDecimals(3)
        self.bp_b_fwhm_lateral.setValue(2.277)
        self.bp_b_fwhm_lateral.setSuffix(" pix")

        self.bp_b_fwhm_axial = QDoubleSpinBox()
        self.bp_b_fwhm_axial.setRange(0.1, 500.0)
        self.bp_b_fwhm_axial.setDecimals(3)
        self.bp_b_fwhm_axial.setValue(7.385)
        self.bp_b_fwhm_axial.setSuffix(" pix")

        bp_gaussian_layout.addRow("A: FWHM, Lateral:", self.bp_a_fwhm_lateral)
        bp_gaussian_layout.addRow("A: FWHM, Axial:", self.bp_a_fwhm_axial)
        bp_gaussian_layout.addRow("B: FWHM, Lateral:", self.bp_b_fwhm_lateral)
        bp_gaussian_layout.addRow("B: FWHM, Axial:", self.bp_b_fwhm_axial)
        self.backprojector_gaussian_widget.setLayout(bp_gaussian_layout)
        self.backprojector_layout.addWidget(self.backprojector_gaussian_widget)

        # Backprojector Custom widgets (hidden by default)
        self.backprojector_custom_widget = QWidget()
        bp_custom_layout = QFormLayout()

        self.bp_a_path = QLineEdit()
        self.bp_a_path.setPlaceholderText("Select Backprojector A file (.npy, .tif)")
        self.bp_a_browse = QPushButton("Browse")
        self.bp_a_browse.clicked.connect(lambda: self.browse_backprojector("a"))

        self.bp_b_path = QLineEdit()
        self.bp_b_path.setPlaceholderText("Select Backprojector B file (.npy, .tif)")
        self.bp_b_browse = QPushButton("Browse")
        self.bp_b_browse.clicked.connect(lambda: self.browse_backprojector("b"))

        bp_a_row = QHBoxLayout()
        bp_a_row.addWidget(self.bp_a_path)
        bp_a_row.addWidget(self.bp_a_browse)

        bp_b_row = QHBoxLayout()
        bp_b_row.addWidget(self.bp_b_path)
        bp_b_row.addWidget(self.bp_b_browse)

        bp_custom_layout.addRow("Backprojector A:", bp_a_row)
        bp_custom_layout.addRow("Backprojector B:", bp_b_row)
        self.backprojector_custom_widget.setLayout(bp_custom_layout)
        self.backprojector_custom_widget.setVisible(False)
        self.backprojector_layout.addWidget(self.backprojector_custom_widget)

        self.backprojector_group.setLayout(self.backprojector_layout)
        self.backprojector_group.setVisible(False)
        psf_layout.addWidget(self.backprojector_group)

        # Show PSF button
        self.show_psf_button = QPushButton("Show PSF")
        self.show_psf_button.clicked.connect(self.show_psf)
        psf_layout.addWidget(self.show_psf_button)

        psf_group.setLayout(psf_layout)

        # === Deconvolution Parameters Section ===
        params_group = QGroupBox("Deconvolution Parameters")
        params_layout = QVBoxLayout()

        # Main parameters
        main_params_layout = QFormLayout()

        self.function_combo = QComboBox()
        self.function_combo.addItems(["Additive", "Eff. Bayes", "diSPIM"])

        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 1000)
        self.iterations_spin.setValue(10)

        self.epsilon_spin = QDoubleSpinBox()
        self.epsilon_spin.setRange(0.0001, 1.0)
        self.epsilon_spin.setValue(0.001)
        self.epsilon_spin.setDecimals(4)

        self.require_both_check = QCheckBox("Require Both")
        self.require_both_check.setChecked(True)

        main_params_layout.addRow("Function:", self.function_combo)
        main_params_layout.addRow("Iterations:", self.iterations_spin)
        main_params_layout.addRow("Epsilon:", self.epsilon_spin)
        main_params_layout.addRow("Require Both:", self.require_both_check)
        params_layout.addLayout(main_params_layout)

        # Boundary Correction subsection
        boundary_group = QGroupBox("Boundary Correction")
        boundary_layout = QVBoxLayout()

        self.boundary_enable_check = QCheckBox("Enable")
        self.boundary_enable_check.toggled.connect(self.on_boundary_toggled)
        boundary_layout.addWidget(self.boundary_enable_check)

        self.boundary_params_widget = QWidget()
        boundary_params = QFormLayout()
        self.boundary_threshold_spin = QDoubleSpinBox()
        self.boundary_threshold_spin.setRange(0.001, 1.0)
        self.boundary_threshold_spin.setValue(0.01)
        self.boundary_threshold_spin.setDecimals(3)
        self.boundary_threshold_spin.setEnabled(False)

        boundary_params.addRow("Significance Threshold:", self.boundary_threshold_spin)
        self.boundary_params_widget.setLayout(boundary_params)
        boundary_layout.addWidget(self.boundary_params_widget)
        boundary_group.setLayout(boundary_layout)
        params_layout.addWidget(boundary_group)

        # Chunkwise subsection
        chunkwise_group = QGroupBox("Chunkwise")
        chunkwise_layout = QVBoxLayout()

        self.chunkwise_enable_check = QCheckBox("Enable")
        self.chunkwise_enable_check.toggled.connect(self.on_chunkwise_toggled)
        chunkwise_layout.addWidget(self.chunkwise_enable_check)

        self.chunkwise_params_widget = QWidget()
        chunk_params_layout = QFormLayout()

        chunk_size_layout = QHBoxLayout()
        chunk_size_layout.addWidget(QLabel("Z:"))
        self.chunk_size_z = QSpinBox()
        self.chunk_size_z.setRange(16, 2048)
        self.chunk_size_z.setValue(64)
        chunk_size_layout.addWidget(self.chunk_size_z)
        chunk_size_layout.addWidget(QLabel("Y:"))
        self.chunk_size_y = QSpinBox()
        self.chunk_size_y.setRange(16, 2048)
        self.chunk_size_y.setValue(128)
        chunk_size_layout.addWidget(self.chunk_size_y)
        chunk_size_layout.addWidget(QLabel("X:"))
        self.chunk_size_x = QSpinBox()
        self.chunk_size_x.setRange(16, 2048)
        self.chunk_size_x.setValue(128)
        chunk_size_layout.addWidget(self.chunk_size_x)

        overlap_layout = QHBoxLayout()
        overlap_layout.addWidget(QLabel("Z:"))
        self.overlap_z = QSpinBox()
        self.overlap_z.setRange(0, 1024)
        self.overlap_z.setValue(40)
        overlap_layout.addWidget(self.overlap_z)
        overlap_layout.addWidget(QLabel("Y:"))
        self.overlap_y = QSpinBox()
        self.overlap_y.setRange(0, 1024)
        self.overlap_y.setValue(40)
        overlap_layout.addWidget(self.overlap_y)
        overlap_layout.addWidget(QLabel("X:"))
        self.overlap_x = QSpinBox()
        self.overlap_x.setRange(0, 1024)
        self.overlap_x.setValue(40)
        overlap_layout.addWidget(self.overlap_x)

        chunk_params_layout.addRow("Chunk Size:", chunk_size_layout)
        chunk_params_layout.addRow("Overlap:", overlap_layout)
        self.chunkwise_params_widget.setLayout(chunk_params_layout)
        chunkwise_layout.addWidget(self.chunkwise_params_widget)
        chunkwise_group.setLayout(chunkwise_layout)
        params_layout.addWidget(chunkwise_group)

        params_group.setLayout(params_layout)

        # === Save/Action Section ===
        action_layout = QVBoxLayout()

        # Save checkbox with path input
        save_row = QHBoxLayout()
        self.save_check = QCheckBox("Save")
        self.save_check.toggled.connect(self.on_save_toggled)
        save_row.addWidget(self.save_check)
        save_row.addStretch()

        self.save_path_widget = QWidget()
        save_path_layout = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        self.save_path_edit.setPlaceholderText("Output path...")
        self.save_path_browse = QPushButton("Browse")
        self.save_path_browse.clicked.connect(self.browse_save_path)
        save_path_layout.addWidget(self.save_path_edit)
        save_path_layout.addWidget(self.save_path_browse)
        self.save_path_widget.setLayout(save_path_layout)
        self.save_path_widget.setVisible(False)
        save_row.addWidget(self.save_path_widget)

        action_layout.addLayout(save_row)

        # Save Only checkbox
        self.save_only_check = QCheckBox("Save Only")
        self.save_only_check.setEnabled(False)
        action_layout.addWidget(self.save_only_check)

        # Deconvolve button
        self.deconvolve_button = QPushButton("Deconvolve")
        self.deconvolve_button.clicked.connect(self.deconvolve_data)
        action_layout.addWidget(self.deconvolve_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Configure deconvolution parameters")
        action_layout.addWidget(self.status_label)

        # Add all sections to main layout
        layout.addWidget(input_group)
        layout.addWidget(psf_group)
        layout.addWidget(params_group)
        layout.addLayout(action_layout)

        self.setLayout(layout)

        # Update layer lists when viewer layers change
        self.viewer.layers.events.inserted.connect(self.update_layer_lists)
        self.viewer.layers.events.removed.connect(self.update_layer_lists)

    def on_input_mode_changed(self):
        """Toggle between Layers and Paths input mode."""
        use_layers = self.input_mode_layers.isChecked()
        self.layers_widget.setVisible(use_layers)
        self.paths_widget.setVisible(not use_layers)

    def on_psf_type_changed(self):
        """Toggle between Gaussian and Custom PSF type."""
        use_gaussian = self.psf_type_combo.currentText() == "Gaussian"
        self.gaussian_widget.setVisible(use_gaussian)
        self.custom_psf_widget.setVisible(not use_gaussian)
        # Also toggle backprojector type visibility
        self.backprojector_gaussian_widget.setVisible(use_gaussian)
        self.backprojector_custom_widget.setVisible(not use_gaussian)

    def on_flipped_psf_toggled(self, checked):
        """Handle 'Backprojector is Flipped PSF' checkbox toggle."""
        self.backprojector_group.setVisible(not checked)

    def on_save_toggled(self, checked):
        """Handle Save checkbox toggle."""
        self.save_path_widget.setVisible(checked)
        self.save_only_check.setEnabled(checked)
        if not checked:
            self.save_only_check.setChecked(False)

    def on_boundary_toggled(self, checked):
        """Handle Boundary Correction Enable checkbox toggle."""
        self.boundary_threshold_spin.setEnabled(checked)

    def on_chunkwise_toggled(self, checked):
        """Handle Chunkwise Enable checkbox toggle."""
        self.chunk_size_z.setEnabled(checked)
        self.chunk_size_y.setEnabled(checked)
        self.chunk_size_x.setEnabled(checked)
        self.overlap_z.setEnabled(checked)
        self.overlap_y.setEnabled(checked)
        self.overlap_x.setEnabled(checked)

    def show_psf(self):
        """Generate PSF volumes and display them as napari layers for inspection."""
        try:
            psf_a, psf_b, _, _ = self._get_psf_and_backprojectors()

            # Remove existing PSF layers if they exist
            for layer_name in ["PSF: View A", "PSF: View B"]:
                try:
                    self.viewer.layers.remove(self.viewer.layers[layer_name])
                except (KeyError, ValueError):
                    pass

            # Add PSF layers
            self.viewer.add_image(psf_a, name="PSF: View A")
            self.viewer.add_image(psf_b, name="PSF: View B")

            self.status_label.setText("PSF layers added: PSF: View A, PSF: View B")

        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate PSF: {e}")
            import traceback
            traceback.print_exc()

    def browse_path(self, view):
        """Browse for zarr directory path."""
        path = QFileDialog.getExistingDirectory(self, f"Select zarr directory for View {view.upper()}")
        if path:
            if view == "a":
                self.path_a_edit.setText(path)
            else:
                self.path_b_edit.setText(path)

    def browse_psf(self, psf_type):
        """Browse for PSF file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select PSF {psf_type.upper()} file", "", "NumPy/TIFF files (*.npy *.tif)"
        )
        if file_path:
            if psf_type == "a":
                self.psf_a_path.setText(file_path)
            else:
                self.psf_b_path.setText(file_path)

    def browse_backprojector(self, bp_type):
        """Browse for backprojector file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"Select Backprojector {bp_type.upper()} file", "", "NumPy/TIFF files (*.npy *.tif)"
        )
        if file_path:
            if bp_type == "a":
                self.bp_a_path.setText(file_path)
            else:
                self.bp_b_path.setText(file_path)

    def browse_save_path(self):
        """Browse for save path."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if path:
            self.save_path_edit.setText(path)

    def update_layer_lists(self, event=None):
        """Update the layer selection dropdowns."""
        # Store current selections
        current_a = self.layer_a_combo.currentText()
        current_b = self.layer_b_combo.currentText()

        # Clear and repopulate
        self.layer_a_combo.clear()
        self.layer_b_combo.clear()

        # Add placeholder items
        self.layer_a_combo.addItem("Select View A layer...")
        self.layer_b_combo.addItem("Select View B layer...")

        # Add available layers
        for layer in self.viewer.layers:
            if hasattr(layer, "data") and layer.data is not None:
                self.layer_a_combo.addItem(layer.name)
                self.layer_b_combo.addItem(layer.name)

        # Restore selections if they still exist
        if current_a and current_a in [self.layer_a_combo.itemText(i) for i in range(self.layer_a_combo.count())]:
            self.layer_a_combo.setCurrentText(current_a)
        if current_b and current_b in [self.layer_b_combo.itemText(i) for i in range(self.layer_b_combo.count())]:
            self.layer_b_combo.setCurrentText(current_b)

    def set_input_data(self, data_dict):
        """Receive input data from registration step (placeholder for future wiring)."""
        pass

    # === Helper methods for collecting UI parameters ===

    def _get_input_arrays(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Collect input data based on UI state (Layers or Paths mode)."""
        if self.input_mode_layers.isChecked():
            # Layers mode - get data from selected napari layers
            layer_a_name = self.layer_a_combo.currentText()
            layer_b_name = self.layer_b_combo.currentText()

            if layer_a_name.startswith("Select"):
                raise ValueError("Please select a View A layer")
            if layer_b_name.startswith("Select"):
                raise ValueError("Please select a View B layer")

            layer_a = self.viewer.layers[layer_a_name]
            layer_b = self.viewer.layers[layer_b_name]

            view_a = numpy.asarray(layer_a.data)
            view_b = numpy.asarray(layer_b.data)
        else:
            # Paths mode - load from zarr files
            path_a = self.path_a_edit.text()
            path_b = self.path_b_edit.text()

            if not path_a or not os.path.exists(path_a):
                raise ValueError("Please provide a valid path for View A")
            if not path_b or not os.path.exists(path_b):
                raise ValueError("Please provide a valid path for View B")

            view_a = numpy.asarray(zarr.open(path_a)[:])
            view_b = numpy.asarray(zarr.open(path_b)[:])

        if view_a.shape != view_b.shape:
            raise ValueError(
                f"View A shape {view_a.shape} does not match View B shape {view_b.shape}"
            )

        return view_a, view_b

    def _load_psf_file(self, path: str) -> numpy.ndarray:
        """Load a PSF array from a .npy or .tif file."""
        if path.endswith(".npy"):
            return numpy.load(path)
        else:
            return tifffile.imread(path)

    def _get_psf_and_backprojectors(
        self,
    ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        """Generate or load PSF and backprojector arrays from UI parameters."""
        # Determine theta angles based on deskew method
        deskew_method = self.deskew_method_combo.currentText()
        if deskew_method == "Shear-Warp":
            theta_a = 0.0  # degrees
            theta_b = 90.0  # degrees
        else:  # Orthogonal
            theta_a = 45.0  # degrees
            theta_b = -45.0  # degrees

        psf_type = self.psf_type_combo.currentText()

        # Get PSF arrays
        if psf_type == "Gaussian":
            # Calculate PSF volume size: FWHM * 3, rounded up
            fwhm_a_lat = self.psf_a_fwhm_lateral.value()
            fwhm_a_ax = self.psf_a_fwhm_axial.value()
            fwhm_b_lat = self.psf_b_fwhm_lateral.value()
            fwhm_b_ax = self.psf_b_fwhm_axial.value()

            size_a = math.ceil(max(fwhm_a_ax, fwhm_a_lat) * 3)
            if size_a % 2 == 0:
                size_a += 1
            size_b = math.ceil(max(fwhm_b_ax, fwhm_b_lat) * 3)
            if size_b % 2 == 0:
                size_b += 1
            
            psf_a = make_gaussian_psf_volume(
                fwhm_ax=fwhm_a_ax, fwhm_lat=fwhm_a_lat,
                size_ax=size_a, size_lat=size_a, theta=theta_a
            )
            psf_b = make_gaussian_psf_volume(
                fwhm_ax=fwhm_b_ax, fwhm_lat=fwhm_b_lat,
                size_ax=size_b, size_lat=size_b, theta=theta_b
            )
        else:  # Custom
            path_a = self.psf_a_path.text()
            path_b = self.psf_b_path.text()
            if not path_a or not os.path.exists(path_a):
                raise ValueError("Please provide a valid PSF file for View A")
            if not path_b or not os.path.exists(path_b):
                raise ValueError("Please provide a valid PSF file for View B")
            psf_a = self._load_psf_file(path_a)
            psf_b = self._load_psf_file(path_b)

        # Get backprojector arrays
        if self.flipped_psf_check.isChecked():
            # Backprojector is flipped PSF
            backproj_a = numpy.ascontiguousarray(psf_a[::-1, ::-1, ::-1])
            backproj_b = numpy.ascontiguousarray(psf_b[::-1, ::-1, ::-1])
        else:
            # Custom backprojectors
            if psf_type == "Gaussian":
                bp_fwhm_a_lat = self.bp_a_fwhm_lateral.value()
                bp_fwhm_a_ax = self.bp_a_fwhm_axial.value()
                bp_fwhm_b_lat = self.bp_b_fwhm_lateral.value()
                bp_fwhm_b_ax = self.bp_b_fwhm_axial.value()

                bp_size_a_ax = math.ceil(bp_fwhm_a_ax * 3)
                bp_size_a_lat = math.ceil(bp_fwhm_a_lat * 3)
                bp_size_b_ax = math.ceil(bp_fwhm_b_ax * 3)
                bp_size_b_lat = math.ceil(bp_fwhm_b_lat * 3)

                backproj_a = make_gaussian_psf_volume(
                    fwhm_ax=bp_fwhm_a_ax, fwhm_lat=bp_fwhm_a_lat,
                    size_ax=bp_size_a_ax, size_lat=bp_size_a_lat, theta=theta_a
                )
                backproj_b = make_gaussian_psf_volume(
                    fwhm_ax=bp_fwhm_b_ax, fwhm_lat=bp_fwhm_b_lat,
                    size_ax=bp_size_b_ax, size_lat=bp_size_b_lat, theta=theta_b
                )
            else:  # Custom
                bp_path_a = self.bp_a_path.text()
                bp_path_b = self.bp_b_path.text()
                if not bp_path_a or not os.path.exists(bp_path_a):
                    raise ValueError("Please provide a valid backprojector file for View A")
                if not bp_path_b or not os.path.exists(bp_path_b):
                    raise ValueError("Please provide a valid backprojector file for View B")
                backproj_a = self._load_psf_file(bp_path_a)
                backproj_b = self._load_psf_file(bp_path_b)

        return psf_a, psf_b, backproj_a, backproj_b

    def _get_decon_function(self) -> str:
        """Map UI function selection to backend string."""
        func_map = {
            "Additive": "additive",
            "Eff. Bayes": "efficient",
            "diSPIM": "dispim",
        }
        return func_map[self.function_combo.currentText()]

    # === Signal handlers for DeconvolutionWorker ===

    def on_deconvolution_finished(self, result: dict):
        """Handle successful deconvolution."""
        QTimer.singleShot(0, lambda: self._on_deconvolution_finished_main(result))

    def _on_deconvolution_finished_main(self, result: dict):
        """Add deconvolved result as a napari layer on the main thread."""

        deconv_result = result["result"]
        save_path = result.get("save_path")

        # Add result as a new napari layer
        output_name = "Deconvolved"
        # Remove old layer if it exists
        try:
            self.viewer.layers.remove(self.viewer.layers[output_name])
        except (KeyError, ValueError):
            pass
        try:
            self.viewer.layers.remove(self.viewer.layers["PSF: View A"])
        except (KeyError, ValueError):
            pass
        try:
            self.viewer.layers.remove(self.viewer.layers["PSF: View B"])
        except (KeyError, ValueError):
            pass
        
        self.viewer.add_image(
            deconv_result,
            name=output_name,
        )

        # Update status
        status_msg = f"Deconvolution completed! Result shape: {deconv_result.shape}"
        if save_path:
            status_msg += f" (saved to {save_path})"
        self.status_label.setText(status_msg)

        # Reset UI state
        self.deconvolve_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with deconvolved data
        self.deconvolved.emit({"result": deconv_result})

    def on_deconvolution_error(self, error_msg: str):
        """Handle deconvolution error."""
        QMessageBox.critical(self, "Error", f"Deconvolution failed: {error_msg}")
        self.status_label.setText("Error during deconvolution")
        self.deconvolve_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_deconvolution_progress(self, message: str, percentage: int):
        """Handle progress update from deconvolution worker."""
        self.status_label.setText(message)
        if percentage >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)

    # === Main deconvolution entry point ===

    def deconvolve_data(self):
        """Start deconvolution based on UI parameters."""
        # Check if local computation is possible
        use_remote = (self.remote_client is not None and self.remote_client.is_connected)
        if not self.has_pyspim and not use_remote:
            QMessageBox.warning(
                self, "pyspim Not Available",
                "Local computation requires pyspim, which is not installed.\n\n"
                "Either:\n"
                "1. Connect to a remote server (Tab 0: Remote Connection), or\n"
                "2. Install pyspim: pip install napari-pyspim[full]"
            )
            return

        try:
            # Validate and collect input arrays
            view_a, view_b = self._get_input_arrays()

            # Validate and collect PSF/backprojector arrays
            psf_a, psf_b, backproj_a, backproj_b = self._get_psf_and_backprojectors()

            # Collect deconvolution parameters
            decon_function = self._get_decon_function()
            num_iter = self.iterations_spin.value()
            epsilon = self.epsilon_spin.value()
            req_both = self.require_both_check.isChecked()
            boundary_correction = self.boundary_enable_check.isChecked()
            boundary_sigma = self.boundary_threshold_spin.value()

            # Collect chunkwise parameters
            chunkwise = self.chunkwise_enable_check.isChecked()
            chunk_size = (
                self.chunk_size_z.value(),
                self.chunk_size_y.value(),
                self.chunk_size_x.value(),
            )
            overlap = (
                self.overlap_z.value(),
                self.overlap_y.value(),
                self.overlap_x.value(),
            )

            # Collect save parameters
            save_path = None
            if self.save_check.isChecked():
                save_path = self.save_path_edit.text()
                if not save_path:
                    QMessageBox.warning(
                        self, "Error", "Please specify an output path for saving"
                    )
                    return

            # Update UI state
            self.deconvolve_button.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            self.status_label.setText("Starting deconvolution...")

            # Create and start worker
            use_remote = (self.remote_client is not None and self.remote_client.is_connected)
            self.decon_worker = DeconvolutionWorker(
                view_a=view_a,
                view_b=view_b,
                psf_a=psf_a,
                psf_b=psf_b,
                backproj_a=backproj_a,
                backproj_b=backproj_b,
                decon_function=decon_function,
                num_iter=num_iter,
                epsilon=epsilon,
                req_both=req_both,
                boundary_correction=boundary_correction,
                boundary_sigma=boundary_sigma,
                chunkwise=chunkwise,
                chunk_size=chunk_size,
                overlap=overlap,
                save_path=save_path,
                use_remote=use_remote,
                remote_client=self.remote_client,
            )
            self.decon_worker.finished.connect(self.on_deconvolution_finished)
            self.decon_worker.error_occurred.connect(self.on_deconvolution_error)
            self.decon_worker.progress_updated.connect(self.on_deconvolution_progress)
            self.decon_worker.start()

        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start deconvolution: {e}")
            import traceback
            traceback.print_exc()
