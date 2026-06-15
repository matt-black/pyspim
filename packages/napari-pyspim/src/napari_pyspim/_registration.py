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
from qtpy.QtCore import QThread, QTimer, Signal
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

from ._remote_client import RemoteClient
from ._sftp_browser import SftpBrowserDialog
from ._utils import decompose_affine_matrix

# Lazy imports to avoid CUDA compilation at module level
# from pyspim.reg import pcc, opt
# from pyspim.interp import affine
# from pyspim.util import pad_to_same_size, launch_params_for_volume

class LoadDeskewWorker(QThread):
    """Worker thread for loading data, deskewing, and computing projections."""

    ready = Signal(dict)
    error_occurred = Signal(str)
    progress_updated = Signal(str)

    def __init__(
        self,
        data_path: str,
        channel: int,
        projection_type: str,
        pixel_size: float,
        step_size: float,
        method: str = "orthogeo",
        ignore_bbox: bool = False,
        multi_pos: bool = False,
        time: int = 0,
        position: int = 0,
        auto_crop: bool = True,
        camera_offset: int = 0,
        psf_fwhm_axial: float = None,
        psf_fwhm_lateral: float = None,
        psf_model: str = "gaussian",
        use_psf_in_plane: bool = False,
        force_cpu: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.channel = channel
        self.projection_type = projection_type
        self.pixel_size = pixel_size
        self.step_size = step_size
        self.theta = math.pi / 4  # Hardcoded to 45 degrees
        self.method = method
        self.ignore_bbox = ignore_bbox
        self.multi_pos = multi_pos
        self.time = time
        self.position = position
        self.auto_crop = auto_crop
        self.camera_offset = camera_offset
        self.psf_fwhm_axial = psf_fwhm_axial
        self.psf_fwhm_lateral = psf_fwhm_lateral
        self.psf_model = psf_model
        self.use_psf_in_plane = use_psf_in_plane
        self.force_cpu = force_cpu

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
            yx_proj = cp.max(volume, axis=0)  # shape: (Y, X)
            zy_proj = cp.max(volume, axis=2)  # shape: (Z, Y)
            xz_proj = cp.max(volume, axis=1)  # shape: (Z, X)
        elif proj_type == "sum":
            yx_proj = cp.sum(volume, axis=0)
            zy_proj = cp.sum(volume, axis=2)
            xz_proj = cp.sum(volume, axis=1)
        elif proj_type == "mean":
            yx_proj = cp.mean(volume, axis=0)
            zy_proj = cp.mean(volume, axis=2)
            xz_proj = cp.mean(volume, axis=1)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        try:
            yx_proj = yx_proj.get()
            zy_proj = zy_proj.get()
            xz_proj = xz_proj.get()
        except:
            pass
        return yx_proj, zy_proj, xz_proj

    @property
    def use_remote(self):
        """Whether to use remote execution. Defaults to False."""
        return getattr(self, '_use_remote', False)

    def run(self):
        """Load data, deskew, and compute projections in background thread."""
        if self.use_remote:
            self._run_remote()
        else:
            self._run_local()

    def _run_local(self):
        """Load data, deskew, and compute projections locally."""
        try:
            from pyspim.data import dispim as data
            from pyspim import deskew as dsk

            # Use numpy for data loading when force_cpu is True
            if self.force_cpu:
                import numpy as load_np
            else:
                try:
                    import cupy as load_np
                except ImportError:
                    import numpy as load_np

            # Load bounding box if present
            window = self._load_bbox()
            print(window)

            # Load data
            self.progress_updated.emit("Loading data...")
            with data.uManagerAcquisition(self.data_path, self.multi_pos, load_np) as acq:
                if self.multi_pos:
                    volume_a = acq.get(self.position, "a", self.channel, self.time, window=window)
                    volume_b = acq.get(self.position, "b", self.channel, self.time, window=window)
                else:
                    volume_a = acq.get("a", self.channel, self.time, window=window)
                    volume_b = acq.get("b", self.channel, self.time, window=window)

            # Subtract camera offset if non-zero
            if self.camera_offset > 0:
                self.progress_updated.emit(f"Subtracting camera offset: {self.camera_offset}...")
                from pyspim.data.dispim import subtract_constant_uint16arr
                volume_a = subtract_constant_uint16arr(volume_a, self.camera_offset)
                volume_b = subtract_constant_uint16arr(volume_b, self.camera_offset)

            self.progress_updated.emit(
                f"Data loaded - A: {volume_a.shape}, B: {volume_b.shape}"
            )

            # Deskew View A (direction = 1)
            self.progress_updated.emit("Deskewing View A...")
            if self.method == "shear":
                kwargs = {
                    "rotation_thetas": (0, 0, 0),
                    "interp_method": "cubspl",
                    "auto_crop": self.auto_crop,
                    "preserve_dtype": True,
                    "block_size": (8,8,8),
                }
            elif self.method == "orthopsf":
                kwargs = {
                    "preserve_dtype": True,
                    "stream": None,
                    "psf_fwhm_axial": self.psf_fwhm_axial,
                    "psf_fwhm_lateral": self.psf_fwhm_lateral,
                    "psf_model": self.psf_model,
                    "use_psf_in_plane": self.use_psf_in_plane,
                }
            elif self.method == "ortho":
                kwargs = {
                    "preserve_dtype": True,
                    "stream": None,
                }
            elif self.method == "affine":
                kwargs = {
                    "preserve_dtype": True,
                    "interp_method": "cubspl",
                    "block_size": (8,8,8),
                }
            else:
                kwargs = {}
            a_dsk = dsk.deskew_stage_scan(
                volume_a, self.pixel_size, self.step_size, 1,
                theta=(self.theta / (math.pi/180.0)),
                method=self.method,
                **kwargs,
            )
            
            # Deskew View B (direction = -1)
            self.progress_updated.emit("Deskewing View B...")
            if self.method == "shear":
                kwargs["rotation_thetas"] = (0, math.pi/2, 0)
            b_dsk = dsk.deskew_stage_scan(
                volume_b, self.pixel_size, self.step_size, -1,
                theta=(self.theta / (math.pi/180.0)),
                method=self.method,
                **kwargs,
            )
            if self.method == "dispim":
                # need to rotate the volume in the XZ plane so that it is in 
                # the same coordinate system as that of view A
                from pyspim.interp.affine import transform as affine_transform
                from pyspim._matrix import rotation_about_point_matrix
                R = rotation_about_point_matrix(
                    0, math.pi/2, 0,
                    *[s/2 for s in b_dsk.shape[::-1]]
                )
                b_dsk = affine_transform(
                    b_dsk, 
                    R,
                    "cubspl",
                    True,
                    (b_dsk.shape[0], b_dsk.shape[1], b_dsk.shape[2]),
                    8, 8, 8
                )
            
            self.progress_updated.emit(f"Deskewing completed")

            # Compute projections
            self.progress_updated.emit("Computing projections...")
            yx_proj_a, zy_proj_a, xz_proj_a = self._compute_projections(a_dsk, self.projection_type)
            yx_proj_b, zy_proj_b, xz_proj_b = self._compute_projections(b_dsk, self.projection_type)

            # move deskewed volumes to cpu, if necessary
            try:
                a_dsk = a_dsk.get()
            except:
                pass

            try:
                b_dsk = b_dsk.get()
            except:
                pass
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
                "step_size": self.step_size,
            }

            self.ready.emit(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _run_remote(self):
        """Load data, deskew, and compute projections via remote server."""
        try:
            params = {
                "data_path": self.data_path,
                "channel": self.channel,
                "projection_type": self.projection_type,
                "pixel_size": self.pixel_size,
                "step_size": self.step_size,
                "theta": self.theta,
                "method": self.method,
                "ignore_bbox": self.ignore_bbox,
                "multi_pos": self.multi_pos,
                "time": self.time,
                "position": self.position,
                "auto_crop": self.auto_crop,
                "camera_offset": self.camera_offset,
                "force_cpu": self.force_cpu,
            }
            # Add orthopsf-specific params if method is orthopsf
            if self.method == "orthopsf":
                params.update({
                    "psf_fwhm_axial": self.psf_fwhm_axial,
                    "psf_fwhm_lateral": self.psf_fwhm_lateral,
                    "psf_model": self.psf_model,
                    "use_psf_in_plane": self.use_psf_in_plane,
                })
            result = self.remote_client.send_command_blocking("load_deskew", params)
            # Convert shape lists to tuples for compatibility
            if "volume_shape_a" in result and isinstance(result["volume_shape_a"], list):
                result["volume_shape_a"] = tuple(result["volume_shape_a"])
            if "volume_shape_b" in result and isinstance(result["volume_shape_b"], list):
                result["volume_shape_b"] = tuple(result["volume_shape_b"])
            self.ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class RegistrationWorker(QThread):
    """Worker thread for registration."""

    registered = Signal(dict)
    error_occurred = Signal(str)
    progress_updated = Signal(str)

    def __init__(
        self,
        a_deskewed,
        b_deskewed,
        transform_type="t+r+s",
        initial_translation=None,
        metric="cr",
        interp_method="cubspl",
        opt_method="powell",
        use_piecewise=True,
        bound_translation=20.0,
        bound_rot_shear=5.0,
        bound_scale=0.05,
        use_remote: bool = False,
        remote_client = None,
    ):
        super().__init__()
        self.a_deskewed = a_deskewed
        self.b_deskewed = b_deskewed
        self.transform_type = transform_type
        self.initial_translation = initial_translation if initial_translation is not None else [0, 0, 0]
        self.metric = metric
        self.interp_method = interp_method
        self.opt_method = opt_method
        self.use_piecewise = use_piecewise
        self.bound_translation = bound_translation
        self.bound_rot_shear = bound_rot_shear
        self.bound_scale = bound_scale
        self.use_remote = use_remote
        self.remote_client = remote_client

    def run(self):
        """Perform registration in background thread."""
        if self.use_remote:
            self._run_remote()
        else:
            self._run_local()

    def _run_local(self):
        """Perform registration locally."""
        try:
            # Lazy imports to avoid CUDA compilation at module level
            from pyspim.interp import affine
            from pyspim.reg import opt
            from pyspim.util import launch_params_for_volume, pad_to_same_size

            shp_a = self.a_deskewed.shape
            shp_b = self.b_deskewed.shape
            # check if volumes are different size, and pad accordingly
            if shp_a[0] != shp_b[0] or shp_a[1] != shp_b[1] or shp_a[2] != shp_b[2]:
                self.progress_updated.emit("Padding volumes to same size...")
                # Pad volumes to same size
                a_dsk, b_dsk = pad_to_same_size(self.a_deskewed, self.b_deskewed)
            else:
                a_dsk, b_dsk = self.a_deskewed, self.b_deskewed
            # Set up initial parameters and bounds
            self.progress_updated.emit("Setting up optimization parameters...")

            # Use initial translation from pre-reg transform (in pixel units), or default to [0, 0, 0]
            t0 = self.initial_translation

            bt = self.bound_translation
            br = self.bound_rot_shear
            bs = self.bound_scale

            if self.transform_type == "t":
                par0 = t0
                bounds = [(t - bt, t + bt) for t in t0]
            elif self.transform_type == "t+r":
                par0 = np.concatenate([t0, np.asarray([0, 0, 0])])
                bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
            elif self.transform_type == "t+r+s":
                par0 = np.concatenate(
                    [t0, np.asarray([0, 0, 0]), np.asarray([1, 1, 1])]
                )
                bounds = (
                    [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3 + [(1 - bs, 1 + bs)] * 3
                )
            elif self.transform_type == "t+sh":
                par0 = np.concatenate(
                    [t0, np.asarray([0, 0, 0, 0, 0, 0])]
                )
                bounds = (
                    [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6
                )
            elif self.transform_type == "t+ssh":
                par0 = np.concatenate(
                    [t0, np.asarray([0, 0, 0])]
                )
                bounds = (
                    [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
                )
            elif self.transform_type == "t+sh+s":
                par0 = np.concatenate(
                    [t0, np.asarray([0, 0, 0, 0, 0, 0]), np.asarray([1, 1, 1])]
                )
                bounds = (
                    [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6 + [(1 - bs, 1 + bs)] * 3
                )

            # Determine launch parameters for GPU
            self.progress_updated.emit("Optimizing registration...")
            launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)

            # Perform optimization
            if self.use_piecewise:
                T, res = opt.optimize_affine_piecewise(
                    cp.asarray(a_dsk),
                    cp.asarray(b_dsk),
                    metric=self.metric,
                    transform=self.transform_type,
                    interp_method=self.interp_method,
                    opt_method=self.opt_method,
                    par0=par0,
                    bounds=bounds,
                    kernel_launch_params=launch_par,
                    verbose=False,
                )
            else:
                T, res = opt.optimize_affine(
                    cp.asarray(a_dsk),
                    cp.asarray(b_dsk),
                    metric=self.metric,
                    transform=self.transform_type,
                    interp_method=self.interp_method,
                    opt_method=self.opt_method,
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
                interp_method=self.interp_method,
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

    def _run_remote(self):
        """Perform registration via remote server."""
        try:
            # For remote mode, send zarr paths instead of arrays
            # The a_deskewed/b_deskewed should be zarr paths from load_deskew
            a_path = self.a_deskewed if isinstance(self.a_deskewed, str) else None
            b_path = self.b_deskewed if isinstance(self.b_deskewed, str) else None

            if a_path is None or b_path is None:
                # Fallback: send arrays directly (for local deskew + remote register)
                self.progress_updated.emit("Sending deskewed volumes to remote server...")
                result = self.remote_client.send_command_blocking("register", {
                    "a_deskewed": self.a_deskewed,
                    "b_deskewed": self.b_deskewed,
                    "transform_type": self.transform_type,
                    "initial_translation": self.initial_translation,
                    "metric": self.metric,
                    "interp_method": self.interp_method,
                    "opt_method": self.opt_method,
                    "use_piecewise": self.use_piecewise,
                    "bound_translation": self.bound_translation,
                    "bound_rot_shear": self.bound_rot_shear,
                    "bound_scale": self.bound_scale,
                })
            else:
                result = self.remote_client.send_command_blocking("register", {
                    "a_zarr_path": a_path,
                    "b_zarr_path": b_path,
                    "transform_type": self.transform_type,
                    "initial_translation": self.initial_translation,
                    "metric": self.metric,
                    "interp_method": self.interp_method,
                    "opt_method": self.opt_method,
                    "use_piecewise": self.use_piecewise,
                    "bound_translation": self.bound_translation,
                    "bound_rot_shear": self.bound_rot_shear,
                    "bound_scale": self.bound_scale,
                })

            # Convert shape lists to tuples for compatibility
            if "shape" in result and isinstance(result["shape"], list):
                result["shape"] = tuple(result["shape"])

            self.registered.emit(result)

        except Exception as e:
            self.error_occurred.emit(str(e))


class ApplyWorker(QThread):
    """Worker thread for applying registration transformations to specified ranges."""

    finished = Signal()
    error_occurred = Signal(str)
    progress_updated = Signal(str, int)  # message, percentage (0-100)

    def __init__(
        self,
        data_path: str,
        params_path: str,
        time_range: tuple,
        channel_range: tuple,
        multi_pos: bool,
        position: int,
        output_folder: str,
        ignore_bbox: bool,
        save_tiffs: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.params_path = params_path
        self.time_range = time_range
        self.channel_range = channel_range
        self.multi_pos = multi_pos
        self.position = position
        self.output_folder = output_folder
        self.ignore_bbox = ignore_bbox
        self.save_tiffs = save_tiffs

    def _load_bbox(self):
        """Load bounding box from bbox_raw.json if it exists and not ignored."""
        if self.ignore_bbox:
            return None
        bbox_path = os.path.join(self.data_path, "bbox_raw.json")
        if not os.path.exists(bbox_path):
            return None
        try:
            with open(bbox_path, "r") as f:
                bbox = json.load(f)
            window = (
                slice(bbox[0][0], bbox[0][1]),
                slice(bbox[1][0], bbox[1][1]),
                slice(bbox[2][0], bbox[2][1]),
            )
            return window
        except (json.JSONDecodeError, IndexError, TypeError):
            return None

    def _get_deskew_kwargs(self, method: str, **extra_kwargs) -> dict:
        """Get kwargs for deskew based on method, matching LoadDeskewWorker logic."""
        if method == "shear":
            return {
                "rotation_thetas": (0, 0, 0),
                "interp_method": "cubspl",
                "auto_crop": False,
                "preserve_dtype": True,
                "block_size": (8, 8, 8),
            }
        elif method == "orthopsf":
            kwargs = {
                "preserve_dtype": True,
                "stream": None,
            }
            for key in ["psf_fwhm_axial", "psf_fwhm_lateral", "psf_model", "use_psf_in_plane"]:
                if key in extra_kwargs:
                    kwargs[key] = extra_kwargs[key]
            return kwargs
        elif method.startswith("ortho"):
            return {
                "preserve_dtype": True,
                "stream": None,
            }
        elif method == "affine":
            return {
                "preserve_dtype": True,
                "interp_method": "cubspl",
                "block_size": (8, 8, 8),
            }
        else:  # dispim
            return {"preserve_dtype": True}

    def run(self):
        """Apply deskewing and registration to specified time/channel ranges."""
        try:
            import math
            import shutil
            import zarr
            import tifffile
            from pyspim.data import dispim as data
            from pyspim import deskew as dsk
            from pyspim.interp import affine

            # Load parameters
            print(f"[ApplyWorker] Loading params from: {self.params_path}")
            with open(self.params_path, "r") as f:
                params = json.load(f)

            dp = params["deskewing_parameters"]
            method = dp["method"]
            step_size = dp["step_size_um"]
            pixel_size = dp["pixel_size_um"]
            camera_offset = dp.get("camera_offset", 0)
            theta_deg = 45.0  # Hardcoded to 45 degrees
            theta_rad = math.pi / 4
            affine_matrix = np.array(params["affine_registration_matrix"])

            # Load registration parameters
            rp = params["registration_parameters"]
            interp_method = rp.get("interp_method", "cubspl")
            pre_reg = params["pre_reg_transform"]
            # Convert pre-reg transform from micrometers to pixel units
            pre_reg_t = [
                pre_reg["tz_um"] / pixel_size,
                pre_reg["ty_um"] / pixel_size,
                pre_reg["tx_um"] / pixel_size,
            ]
            # Load crop bounds from saved params
            crop_bounds = params.get("crop_bounds")

            # Get deskew kwargs
            if method == "orthopsf":
                extra_kwargs = {
                    k: dp[k] for k in ["psf_fwhm_axial", "psf_fwhm_lateral", "psf_model", "use_psf_in_plane"]
                    if k in dp
                }
                deskew_kwargs = self._get_deskew_kwargs(method, **extra_kwargs)
            else:
                deskew_kwargs = self._get_deskew_kwargs(method)

            # Load bbox
            window = self._load_bbox()

            # Prepare output folder
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)
            print(f"[ApplyWorker] Output folder created: {self.output_folder}")

            # Determine channel indices (already 0-indexed)
            chan_start = self.channel_range[0]
            chan_end = self.channel_range[1]
            channels = list(range(chan_start, chan_end + 1))
            n_channels = len(channels)

            time_start, time_end = self.time_range
            total_items = (time_end - time_start + 1) * len(channels)
            current_item = 0

            self.progress_updated.emit(
                f"Starting apply: {time_end - time_start + 1} timepoints x {n_channels} channels", 0
            )

            for t in range(time_start, time_end + 1):
                # Process first channel to determine output shape
                first_chan = channels[0]
                self.progress_updated.emit(
                    f"Processing Time {t}, Channel {first_chan + 1} (determining shape)...", 0
                )

                # Load and process first channel to get output shape
                with data.uManagerAcquisition(self.data_path, self.multi_pos, np) as acq:
                    if self.multi_pos:
                        vol_a = acq.get(self.position, "a", first_chan, t, window=window)
                        vol_b = acq.get(self.position, "b", first_chan, t, window=window)
                    else:
                        vol_a = acq.get("a", first_chan, t, window=window)
                        vol_b = acq.get("b", first_chan, t, window=window)

                # Subtract camera offset if non-zero
                if camera_offset > 0:
                    from pyspim.data.dispim import subtract_constant_uint16arr
                    vol_a = subtract_constant_uint16arr(vol_a, camera_offset)
                    vol_b = subtract_constant_uint16arr(vol_b, camera_offset)

                a_dsk = dsk.deskew_stage_scan(
                    vol_a, pixel_size, step_size, 1,
                    theta=theta_rad, method=method, **deskew_kwargs
                )
                try:
                    a_dsk = a_dsk.get()
                except:
                    pass

                b_dsk = dsk.deskew_stage_scan(
                    vol_b, pixel_size, step_size, -1,
                    theta=theta_rad, method=method, **deskew_kwargs
                )
                try:
                    b_dsk = b_dsk.get()
                except:
                    pass

                # Apply crop bounds if present in saved params
                if crop_bounds:
                    z_start = crop_bounds["z_start"]
                    z_end = crop_bounds["z_end"]
                    y_start = crop_bounds["y_start"]
                    y_end = crop_bounds["y_end"]
                    x_start = crop_bounds["x_start"]
                    x_end = crop_bounds["x_end"]
                    tz, ty, tx = pre_reg_t

                    # Crop View A
                    a_dsk = a_dsk[z_start:z_end, y_start:y_end, x_start:x_end]

                    # Compute corresponding crop in View B coordinates
                    Zb, Yb, Xb = b_dsk.shape
                    z_start_b = max(0, int(z_start - tz))
                    z_end_b = min(Zb, int(z_end - tz))
                    y_start_b = max(0, int(y_start - ty))
                    y_end_b = min(Yb, int(y_end - ty))
                    x_start_b = max(0, int(x_start - tx))
                    x_end_b = min(Xb, int(x_end - tx))
                    b_dsk = b_dsk[z_start_b:z_end_b, y_start_b:y_end_b, x_start_b:x_end_b]

                # Apply affine transform to B
                b_cupy = cp.asarray(b_dsk)
                b_reg = affine.transform(
                    b_cupy,
                    cp.asarray(affine_matrix),
                    interp_method=interp_method,
                    preserve_dtype=True,
                    out_shp=None,
                    block_size_z=8,
                    block_size_y=8,
                    block_size_x=8,
                ).get()
                # Crop to smallest size
                min_shape = tuple(min(a, b) for a, b in zip(a_dsk.shape, b_reg.shape))
                out_shape = (n_channels, *min_shape)

                # Create zarr arrays for this timepoint
                a_zarr_path = os.path.join(self.output_folder, f"a_t{t}.zarr")
                b_zarr_path = os.path.join(self.output_folder, f"b_t{t}.zarr")

                arr_a = zarr.open(
                    a_zarr_path, mode="w", shape=out_shape,
                    dtype=np.uint16, chunks=(1, 64, 256, 256),
                )
                arr_b = zarr.open(
                    b_zarr_path, mode="w", shape=out_shape,
                    dtype=np.uint16, chunks=(1, 64, 256, 256),
                )

                # Write first channel
                a_final = a_dsk[:min_shape[0], :min_shape[1], :min_shape[2]]
                b_final = b_reg[:min_shape[0], :min_shape[1], :min_shape[2]]
                arr_a[0, ...] = a_final.astype(np.uint16)
                arr_b[0, ...] = b_final.astype(np.uint16)

                current_item += 1
                percentage = int((current_item / total_items) * 100)
                self.progress_updated.emit(
                    f"Time {t}, Channel {first_chan} done ({percentage}%)", percentage
                )

                # Helper to save TIFF after all channels are written
                def _save_tiff_if_needed(zarr_path: str, data_arr: "zarr.Array"):
                    from dask.array import from_zarr
                    dask_data = from_zarr(data_arr)
                    if self.save_tiffs:
                        tiff_path = zarr_path.replace(".zarr", ".tiff")
                        tifffile.imwrite(
                            tiff_path,
                            dask_data,
                            bigtiff=True,
                            photometric="minisblack",
                            resolution=(1 / pixel_size, 1 / pixel_size),
                            metadata={"axes": "CZYX", "spacing": pixel_size, "units": "um"},
                            tile=(1024,1024),
                        )

                # Process remaining channels
                for chan_idx in channels[1:]:
                    c = chan_idx - chan_start  # index within the range

                    # Load raw data
                    with data.uManagerAcquisition(self.data_path, self.multi_pos, np) as acq:
                        if self.multi_pos:
                            vol_a = acq.get(self.position, "a", chan_idx, t, window=window)
                            vol_b = acq.get(self.position, "b", chan_idx, t, window=window)
                        else:
                            vol_a = acq.get("a", chan_idx, t, window=window)
                            vol_b = acq.get("b", chan_idx, t, window=window)

                    # Subtract camera offset if non-zero
                    if camera_offset > 0:
                        from pyspim.data.dispim import subtract_constant_uint16arr
                        vol_a = subtract_constant_uint16arr(vol_a, camera_offset)
                        vol_b = subtract_constant_uint16arr(vol_b, camera_offset)

                    # Deskew View A
                    a_dsk = dsk.deskew_stage_scan(
                        vol_a, pixel_size, step_size, 1,
                        theta=theta_rad, method=method, **deskew_kwargs
                    )
                    try:
                        a_dsk = a_dsk.get()
                    except:
                        pass

                    # Deskew View B
                    b_dsk = dsk.deskew_stage_scan(
                        vol_b, pixel_size, step_size, -1,
                        theta=theta_rad, method=method, **deskew_kwargs
                    )
                    try:
                        b_dsk = b_dsk.get()
                    except:
                        pass

                    # Apply crop bounds if present in saved params
                    if crop_bounds:
                        z_start = crop_bounds["z_start"]
                        z_end = crop_bounds["z_end"]
                        y_start = crop_bounds["y_start"]
                        y_end = crop_bounds["y_end"]
                        x_start = crop_bounds["x_start"]
                        x_end = crop_bounds["x_end"]
                        tz, ty, tx = pre_reg_t

                        # Crop View A
                        a_dsk = a_dsk[z_start:z_end, y_start:y_end, x_start:x_end]

                        # Compute corresponding crop in View B coordinates
                        Zb, Yb, Xb = b_dsk.shape
                        z_start_b = max(0, int(z_start - tz))
                        z_end_b = min(Zb, int(z_end - tz))
                        y_start_b = max(0, int(y_start - ty))
                        y_end_b = min(Yb, int(y_end - ty))
                        x_start_b = max(0, int(x_start - tx))
                        x_end_b = min(Xb, int(x_end - tx))
                        b_dsk = b_dsk[z_start_b:z_end_b, y_start_b:y_end_b, x_start_b:x_end_b]

                    # Apply affine transform to B
                    b_cupy = cp.asarray(b_dsk)
                    b_reg = affine.transform(
                        b_cupy,
                        cp.asarray(affine_matrix),
                        interp_method=interp_method,
                        preserve_dtype=True,
                        out_shp=None,
                        block_size_z=8,
                        block_size_y=8,
                        block_size_x=8,
                    )
                    b_reg = b_reg.get()

                    # Crop to the determined shape
                    a_final = a_dsk[:min_shape[0], :min_shape[1], :min_shape[2]]
                    b_final = b_reg[:min_shape[0], :min_shape[1], :min_shape[2]]

                    # Write to zarr arrays
                    arr_a[c, ...] = a_final.astype(np.uint16)
                    arr_b[c, ...] = b_final.astype(np.uint16)

                    current_item += 1
                    percentage = int((current_item / total_items) * 100)
                    self.progress_updated.emit(
                        f"Time {t}, Channel {chan_idx} done ({percentage}%)", percentage
                    )

                # Save TIFF files after all channels for this timepoint are written
                _save_tiff_if_needed(a_zarr_path, arr_a)
                _save_tiff_if_needed(b_zarr_path, arr_b)

            self.progress_updated.emit("Apply completed!", 100)
            self.finished.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))


class RegistrationWidget(QWidget):
    """Widget for registering dual-view data.
    
    Combines data loading, deskewing, projection-based manual alignment,
    and automated registration.
    """

    registered = Signal(dict)

    def __init__(self, viewer, remote_client: RemoteClient | None = None):
        super().__init__()
        self.viewer = viewer
        self.remote_client = remote_client
        self.load_worker = None
        self.reg_worker = None
        self.apply_worker = None
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
        # Registered projection layer references (B only)
        self.projection_yx_b_reg_layer = None
        self.projection_zy_b_reg_layer = None
        self.projection_xz_b_reg_layer = None
        # Cropped View A registered projection layer references
        self.projection_yx_a_reg_layer = None
        self.projection_zy_a_reg_layer = None
        self.projection_xz_a_reg_layer = None
        # Crop shapes layer references (3 layers for YX, ZY, XZ projections)
        # These are the "Apply" ROI layers used for output cropping
        self.shapes_yx_layer = None
        self.shapes_zy_layer = None
        self.shapes_xz_layer = None
        # Registration-specific crop shapes layer references (3 layers for YX, ZY, XZ)
        # These are the "Reg" ROI layers used only during registration
        self.shapes_yx_reg_layer = None
        self.shapes_zy_reg_layer = None
        self.shapes_xz_reg_layer = None
        # Pre-registration transform state (stored in micrometers: [tz, ty, tx])
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        # Flag to prevent recursive sync when programmatically updating layers
        self._syncing_transforms = False
        # Flag to prevent recursive shape updates during cross-view sync (Apply layers)
        self._syncing_shapes = False
        # Flag to prevent recursive shape updates during cross-view sync (Reg layers)
        self._syncing_reg_shapes = False
        # Store pixel size for layer scale
        self._pixel_size = None
        # Store registration results for saving
        self.registration_matrix = None
        self.correlation_ratio = None
        # Remote mode state
        self._remote_mode = False
        self._pending_command_type = None  # 'load_deskew' or 'query_positions'
        self._pending_data_path = None
        self._session_id = None  # Server-side session ID for deskewed volumes
        self._remote_params_saved = False  # Track if params file exists on remote
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

        # Channel selection
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(0, 19)
        self.channel_spin.setValue(0)
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

        self.ignore_bbox_checkbox = QCheckBox("Ignore bounding box")
        self.ignore_bbox_checkbox.setToolTip("If checked, load all data for deskewing instead of using bbox_raw.json")
        path_layout.addRow(self.ignore_bbox_checkbox)

        self.camera_offset_spin = QSpinBox()
        self.camera_offset_spin.setRange(0, 65535)
        self.camera_offset_spin.setValue(100)
        self.camera_offset_spin.setToolTip(
            "Subtract this constant value from raw pixel intensities after loading."
        )
        path_layout.addRow("Camera Offset:", self.camera_offset_spin)

        path_group.setLayout(path_layout)

        # Deskewing parameters
        deskew_group = QGroupBox("Deskewing Parameters")
        deskew_layout = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(
            ["orthogeo", "orthopsf", "dispim", "shear", "affine"]
        )
        self.method_combo.setCurrentText("orthogeo")

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
        
        deskew_layout.addRow("Method:", self.method_combo)
        deskew_layout.addRow("Step Size:", self.step_size_spin)
        deskew_layout.addRow("Pixel Size:", self.pixel_size_spin)

        # Auto-Crop checkbox (only visible for shear method)
        self.auto_crop_checkbox = QCheckBox("Auto-Crop")
        self.auto_crop_checkbox.setToolTip(
            "When using shear deskewing, automatically crop the output "
            "to remove empty borders introduced by the shearing process."
        )
        self.auto_crop_checkbox.setChecked(True)
        deskew_layout.addRow(self.auto_crop_checkbox)

        # PSF Parameters group (visible only for orthopsf method)
        self.psf_group = QGroupBox("PSF Parameters")
        psf_layout = QFormLayout()

        self.psf_fwhm_axial_spin = QDoubleSpinBox()
        self.psf_fwhm_axial_spin.setRange(0.1, 100.0)
        self.psf_fwhm_axial_spin.setValue(1.6744)
        self.psf_fwhm_axial_spin.setDecimals(4)
        self.psf_fwhm_axial_spin.setSuffix(" μm")
        self.psf_fwhm_axial_spin.setToolTip("Axial PSF FWHM in micrometers")
        psf_layout.addRow("Axial FWHM:", self.psf_fwhm_axial_spin)

        self.psf_fwhm_lateral_spin = QDoubleSpinBox()
        self.psf_fwhm_lateral_spin.setRange(0.01, 50.0)
        self.psf_fwhm_lateral_spin.setValue(0.3245)
        self.psf_fwhm_lateral_spin.setDecimals(4)
        self.psf_fwhm_lateral_spin.setSuffix(" μm")
        self.psf_fwhm_lateral_spin.setToolTip("Lateral PSF FWHM in micrometers")
        psf_layout.addRow("Lateral FWHM:", self.psf_fwhm_lateral_spin)

        self.psf_model_combo = QComboBox()
        self.psf_model_combo.addItems(["Gaussian", "Airy", "Lorentzian"])
        self.psf_model_combo.setCurrentText("Gaussian")
        self.psf_model_combo.setToolTip("PSF model for weighting adjacent planes")
        psf_layout.addRow("PSF Model:", self.psf_model_combo)

        self.use_psf_in_plane_checkbox = QCheckBox("Use PSF In-Plane")
        self.use_psf_in_plane_checkbox.setToolTip(
            "When enabled, uses PSF-weighted interpolation within each plane "
            "instead of standard bilinear interpolation."
        )
        self.use_psf_in_plane_checkbox.setChecked(False)
        psf_layout.addRow(self.use_psf_in_plane_checkbox)

        self.psf_group.setLayout(psf_layout)

        # Force CPU checkbox (visible only for ortho* methods)
        self.force_cpu_checkbox = QCheckBox("Force CPU")
        self.force_cpu_checkbox.setToolTip(
            "When enabled, forces CPU computation for orthogeo and orthopsf deskewing. "
            "Useful when GPU memory is insufficient for large volumes."
        )
        self.force_cpu_checkbox.setChecked(False)

        # Connect method combo to show/hide method-specific widgets
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        # Set initial visibility based on default method
        self._on_method_changed()

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
        self.transform_combo.addItems(["t", 
                                       "t+r", "t+sh", "t+ssh", 
                                       "t+r+s", "t+sh+s"])
        self.transform_combo.setCurrentText("t+r+s")

        params_layout.addRow("Transform Type:", self.transform_combo)

        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Norm. Inner Prod.", "Corr. Ratio", "Norm. XCorr"])
        self.metric_combo.setCurrentText("Norm. XCorr")
        params_layout.addRow("Metric:", self.metric_combo)

        self.interp_method_combo = QComboBox()
        self.interp_method_combo.addItems(["Linear", "Cubic Spline"])
        self.interp_method_combo.setCurrentText("Cubic Spline")
        params_layout.addRow("Interpolation Method:", self.interp_method_combo)

        self.opt_method_combo = QComboBox()
        self.opt_method_combo.addItems(["Powell", "L-BFGS-B", "COBYLA", "COBYQA", "Nelder-Mead", "TNC"])
        self.opt_method_combo.setCurrentText("Powell")
        params_layout.addRow("Optimization Method:", self.opt_method_combo)

        # Bounds group
        bounds_group = QGroupBox("Bounds")
        bounds_layout = QFormLayout()

        self.bound_translation_spin = QDoubleSpinBox()
        self.bound_translation_spin.setRange(0.1, 1000)
        self.bound_translation_spin.setValue(20.0)
        self.bound_translation_spin.setDecimals(2)
        self.bound_translation_spin.setSuffix(" px")
        self.bound_translation_spin.setToolTip(
            "Maximum deviation (in pixels) from the initial translation guess "
            "during optimization. The optimizer will search within +/- this value "
            "for each axis."
        )
        bounds_layout.addRow("Translation:", self.bound_translation_spin)

        self.bound_rot_shear_spin = QDoubleSpinBox()
        self.bound_rot_shear_spin.setRange(0.1, 100)
        self.bound_rot_shear_spin.setValue(5.0)
        self.bound_rot_shear_spin.setDecimals(2)
        self.bound_rot_shear_spin.setSuffix(" deg")
        self.bound_rot_shear_spin.setToolTip(
            "Maximum deviation (in degrees) from the initial rotation/shear guess "
            "during optimization. The optimizer will search within +/- this value "
            "for each parameter."
        )
        bounds_layout.addRow("Rotation/Shear:", self.bound_rot_shear_spin)

        self.bound_scale_spin = QDoubleSpinBox()
        self.bound_scale_spin.setRange(0.01, 0.5)
        self.bound_scale_spin.setValue(0.05)
        self.bound_scale_spin.setDecimals(3)
        self.bound_scale_spin.setToolTip(
            "Maximum fractional deviation from the initial scale guess (1.0) "
            "during optimization. The optimizer will search within 1 +/- this "
            "value for each axis."
        )
        bounds_layout.addRow("Scale:", self.bound_scale_spin)

        bounds_group.setLayout(bounds_layout)
        params_layout.addRow(bounds_group)

        self.piecewise_checkbox = QCheckBox("Piecewise Optimization")
        self.piecewise_checkbox.setToolTip(
            "When enabled, performs piecewise optimization, optimizing each "
            "transform component (translation, rotation, scale) sequentially. "
            "When disabled, optimizes all components simultaneously."
        )
        self.piecewise_checkbox.setChecked(True)
        params_layout.addRow(self.piecewise_checkbox)

        params_group.setLayout(params_layout)

        # Register button
        self.register_button = QPushButton("Register Data")
        self.register_button.clicked.connect(self.register_data)
        self.register_button.setEnabled(False)

        # Save button
        self.save_button = QPushButton("Save Parameters")
        self.save_button.clicked.connect(self.save_registration_params)
        self.save_button.setEnabled(False)

        # Apply group (always visible, enabled when params file exists)
        self.apply_group = QGroupBox("Apply")
        apply_layout = QVBoxLayout()

        # Time range
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("Time:"))
        self.time_min_spin = QSpinBox()
        self.time_min_spin.setRange(0, 1000)
        self.time_min_spin.setValue(0)
        self.time_max_spin = QSpinBox()
        self.time_max_spin.setRange(0, 1000)
        self.time_max_spin.setValue(0)
        time_row.addWidget(self.time_min_spin)
        time_row.addWidget(QLabel("-"))
        time_row.addWidget(self.time_max_spin)
        apply_layout.addLayout(time_row)

        # Channel range
        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("Channel:"))
        self.channel_min_spin = QSpinBox()
        self.channel_min_spin.setRange(0, 19)
        self.channel_min_spin.setValue(0)
        self.channel_max_spin = QSpinBox()
        self.channel_max_spin.setRange(0, 19)
        self.channel_max_spin.setValue(0)
        chan_row.addWidget(self.channel_min_spin)
        chan_row.addWidget(QLabel("-"))
        chan_row.addWidget(self.channel_max_spin)
        apply_layout.addLayout(chan_row)

        # Save TIFFs checkbox
        self.save_tiffs_checkbox = QCheckBox("Save TIFFs")
        self.save_tiffs_checkbox.setToolTip(
            "When enabled, saves deskewed volumes as TIFF files in addition to Zarr."
        )
        apply_layout.addWidget(self.save_tiffs_checkbox)

        # Batch Job checkbox
        self.batch_job_checkbox = QCheckBox("Compute as Batch Job")
        self.batch_job_checkbox.setToolTip(
            "Submit apply computation as a SLURM batch job "
            "(requires active remote connection)."
        )
        self.batch_job_checkbox.setEnabled(False)
        apply_layout.addWidget(self.batch_job_checkbox)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_registration)
        self.apply_button.setEnabled(False)
        apply_layout.addWidget(self.apply_button)

        self.apply_group.setLayout(apply_layout)

        # Results info
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)

        # Add widgets to layout
        layout.addWidget(path_group)
        layout.addWidget(deskew_group)
        layout.addWidget(self.psf_group)
        layout.addWidget(self.force_cpu_checkbox)
        layout.addWidget(self.load_deskew_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(pre_reg_group)
        layout.addWidget(params_group)
        layout.addWidget(self.register_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.apply_group)
        layout.addWidget(self.results_label)

        self.setLayout(layout)

        # Connect path validation
        self.path_edit.textChanged.connect(self.validate_path)
        self.path_edit.textChanged.connect(self._update_apply_section)

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
            self.load_deskew_button.setEnabled(bool(path))
        else:
            if path and os.path.exists(path):
                self.load_deskew_button.setEnabled(True)
            else:
                self.load_deskew_button.setEnabled(False)

    def _on_multi_pos_toggled(self, checked: bool):
        """Handle Multi-Position checkbox toggled."""
        self.position_spin.setVisible(checked)
        if checked:
            # Try to auto-detect number of positions from the data
            path = self.path_edit.text()
            if not path:
                return
            if self._remote_mode and self.remote_client:
                # Query position count on the remote server
                self._query_positions_remote(path)
            elif os.path.exists(path):
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

    def _is_remote_connected(self):
        """Check if remote connection is active."""
        return self.remote_client is not None and self.remote_client.is_connected

    def _on_method_changed(self, method: str = ""):
        """Handle Method combo box changed - show/hide method-specific widgets."""
        current_method = method if method else self.method_combo.currentText()
        self.auto_crop_checkbox.setVisible(current_method == "shear")
        # Show PSF parameters only for orthopsf method
        self.psf_group.setVisible(current_method == "orthopsf")
        # Set default FWHM values when switching to orthopsf
        if current_method == "orthopsf":
            self.psf_fwhm_axial_spin.setValue(2.1)
            self.psf_fwhm_lateral_spin.setValue(0.381)
        # Enable Force CPU only for ortho* methods
        if current_method in ("orthogeo", "orthopsf"):
            self.force_cpu_checkbox.setEnabled(True)
        else:
            self.force_cpu_checkbox.setEnabled(False)
            self.force_cpu_checkbox.setChecked(False)

    def _map_psf_model(self, display_name: str) -> str:
        """Map PSF model display name to internal string."""
        model_map = {
            "Gaussian": "gaussian",
            "Airy": "airy",
            "Lorentzian": "lorentzian",
        }
        return model_map.get(display_name, "gaussian")

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
        if self.projection_yx_b_reg_layer:
            layers_to_remove.append(self.projection_yx_b_reg_layer)
        if self.projection_zy_b_reg_layer:
            layers_to_remove.append(self.projection_zy_b_reg_layer)
        if self.projection_xz_b_reg_layer:
            layers_to_remove.append(self.projection_xz_b_reg_layer)
        if self.projection_yx_a_reg_layer:
            layers_to_remove.append(self.projection_yx_a_reg_layer)
        if self.projection_zy_a_reg_layer:
            layers_to_remove.append(self.projection_zy_a_reg_layer)
        if self.projection_xz_a_reg_layer:
            layers_to_remove.append(self.projection_xz_a_reg_layer)
        # Crop shapes layers (Apply ROI)
        if self.shapes_yx_layer:
            layers_to_remove.append(self.shapes_yx_layer)
        if self.shapes_zy_layer:
            layers_to_remove.append(self.shapes_zy_layer)
        if self.shapes_xz_layer:
            layers_to_remove.append(self.shapes_xz_layer)
        # Registration crop shapes layers (Reg ROI)
        if self.shapes_yx_reg_layer:
            layers_to_remove.append(self.shapes_yx_reg_layer)
        if self.shapes_zy_reg_layer:
            layers_to_remove.append(self.shapes_zy_reg_layer)
        if self.shapes_xz_reg_layer:
            layers_to_remove.append(self.shapes_xz_reg_layer)

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
        self.projection_yx_b_reg_layer = None
        self.projection_zy_b_reg_layer = None
        self.projection_xz_b_reg_layer = None
        self.projection_yx_a_reg_layer = None
        self.projection_zy_a_reg_layer = None
        self.projection_xz_a_reg_layer = None
        # Reset crop shapes layers (Apply ROI)
        self.shapes_yx_layer = None
        self.shapes_zy_layer = None
        self.shapes_xz_layer = None
        # Reset registration crop shapes layers (Reg ROI)
        self.shapes_yx_reg_layer = None
        self.shapes_zy_reg_layer = None
        self.shapes_xz_reg_layer = None

        # Reset pre-reg transform state
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        self._syncing_transforms = False
        self._syncing_shapes = False
        self._syncing_reg_shapes = False
        self._pixel_size = None
        self.reset_transform_button.setEnabled(False)
        self._update_pre_reg_label()
        # Reset remote session
        self._session_id = None
        # Disable apply section when loading new data
        self._set_apply_enabled(False)

    def load_and_deskew(self):
        """Load data, deskew, and display projections.

        Uses remote server when connection is active and path is remote,
        otherwise falls back to local execution.
        """
        data_path = self.path_edit.text()
        channel = self.channel_spin.value()
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

        # Reset stored data
        self.a_deskewed = None
        self.b_deskewed = None

        self.load_deskew_button.setEnabled(False)
        self.register_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self._set_apply_enabled(False)
        self.registration_matrix = None
        self.correlation_ratio = None
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Loading data and deskewing...")
        self.results_label.setText("")

        # Get deskewing parameters
        pixel_size = self.pixel_size_spin.value()
        step_size = self.step_size_spin.value()
        theta_deg = 45.0  # Hardcoded to 45 degrees
        theta_rad = math.pi / 4
        method = self.method_combo.currentText()
        auto_crop = self.auto_crop_checkbox.isChecked()
        ignore_bbox = self.ignore_bbox_checkbox.isChecked()
        camera_offset = self.camera_offset_spin.value()

        # Branch to local or remote execution
        if self._remote_mode and self.remote_client:
            self._load_deskew_remote(
                data_path, channel, projection_type,
                pixel_size, step_size,
                method, ignore_bbox,
                multi_pos, time, position,
                auto_crop, camera_offset,
            )
        else:
            self._load_deskew_local(
                data_path, channel, projection_type,
                pixel_size, step_size,
                method, ignore_bbox,
                multi_pos, time, position,
                auto_crop, camera_offset,
            )

    def _load_deskew_local(
        self, data_path, channel, projection_type,
        pixel_size, step_size,
        method, ignore_bbox,
        multi_pos, time, position,
        auto_crop, camera_offset,
    ):
        """Load deskew locally using LoadDeskewWorker."""
        psf_fwhm_axial = self.psf_fwhm_axial_spin.value() if method == "orthopsf" else None
        psf_fwhm_lateral = self.psf_fwhm_lateral_spin.value() if method == "orthopsf" else None
        psf_model = self._map_psf_model(self.psf_model_combo.currentText()) if method == "orthopsf" else "gaussian"
        use_psf_in_plane = self.use_psf_in_plane_checkbox.isChecked() if method == "orthopsf" else False
        force_cpu = self.force_cpu_checkbox.isChecked()

        self.load_worker = LoadDeskewWorker(
            data_path, channel, projection_type,
            pixel_size, step_size,
            method, ignore_bbox,
            multi_pos, time, position,
            auto_crop, camera_offset,
            psf_fwhm_axial, psf_fwhm_lateral, psf_model, use_psf_in_plane,
            force_cpu,
        )
        self.load_worker.ready.connect(self.on_load_deskew_ready)
        self.load_worker.error_occurred.connect(self.on_error)
        self.load_worker.progress_updated.connect(self.update_progress)
        self.load_worker.start()

    def _load_deskew_remote(
        self, data_path, channel, projection_type,
        pixel_size, step_size,
        method, ignore_bbox,
        multi_pos, time, position,
        auto_crop, camera_offset,
    ):
        """Load deskew via remote server using signal-based pattern."""
        if not self.remote_client:
            self.on_error("Remote client not available")
            return

        params = {
            "data_path": data_path,
            "channel": channel,
            "projection_type": projection_type,
            "pixel_size": pixel_size,
            "step_size": step_size,
            "theta": math.pi / 4,  # Hardcoded to 45 degrees
            "method": method,
            "ignore_bbox": ignore_bbox,
            "multi_pos": multi_pos,
            "time": time,
            "position": position,
            "auto_crop": auto_crop,
            "camera_offset": camera_offset,
            "force_cpu": self.force_cpu_checkbox.isChecked(),
        }

        if method == "orthopsf":
            params.update({
                "psf_fwhm_axial": self.psf_fwhm_axial_spin.value(),
                "psf_fwhm_lateral": self.psf_fwhm_lateral_spin.value(),
                "psf_model": self._map_psf_model(self.psf_model_combo.currentText()),
                "use_psf_in_plane": self.use_psf_in_plane_checkbox.isChecked(),
            })

        # Store context for signal handler
        self._pending_command_type = "load_deskew"
        self._pending_data_path = data_path

        try:
            # Use callback=None — rely on command_response signal (marshaled to main thread)
            self.remote_client.send_command(
                command="load_deskew",
                params=params,
                callback=None,
                progress_callback=None,
            )
        except Exception as e:
            self.on_error(f"Failed to send command: {e}")

    def on_load_deskew_ready(self, result):
        """Handle successful load and deskew - schedule layer creation on main thread."""
        QTimer.singleShot(0, lambda: self._on_load_deskew_ready_main(result))

    def _on_load_deskew_ready_main(self, result):
        """Add projection layers on main thread."""
        try:
            # Store deskewed data for registration
            self.a_deskewed = result.get("a_deskewed")
            self.b_deskewed = result.get("b_deskewed")

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
                opacity=0.9,
                blending="additive",
                scale=(pixel_size, pixel_size),
            )

            # Add YX projection layer View B (transposed: X, Y) - same coordinates as A
            self.projection_yx_b_layer = self.viewer.add_image(
                yx_proj_b_t,
                name="YX Projection (View B)",
                colormap="cyan",
                opacity=0.9,
                blending="additive",
                scale=(pixel_size, pixel_size),
            )

            # Add ZY projection layer View A (Z, Y) - offset vertically (above) YX
            self.projection_zy_a_layer = self.viewer.add_image(
                zy_proj_a,
                name="ZY Projection (View A)",
                colormap="red",
                opacity=0.9,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(offset_z_um, 0),
            )

            # Add ZY projection layer View B (Z, Y) - same offset as A
            self.projection_zy_b_layer = self.viewer.add_image(
                zy_proj_b,
                name="ZY Projection (View B)",
                colormap="cyan",
                opacity=0.9,
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
                opacity=0.9,
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
                opacity=0.9,
                blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(0, offset_x_um),
            )

            # --- Add crop shapes layers ---
            # Two sets of ROI layers:
            # 1. "Reg" layers (orange) - used for registration cropping
            # 2. Apply layers (lime) - used for output cropping in Apply
            # Both start identical at full projection bounds.
            # Reg ROI must always be within Apply ROI bounds.

            # YX shapes layer (Reg ROI): display (X, Y), row=X, col=Y
            X, Y = yx_proj_a_t.shape
            yx_rect = [[(0, 0), (0, Y - 1), (X - 1, Y - 1), (X - 1, 0)]]
            self.shapes_yx_reg_layer = self.viewer.add_shapes(
                yx_rect,
                shape_type="rectangle",
                name="Reg. YX Crop ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_yx_a_layer.translate,
            )
            self.shapes_yx_reg_layer.editable = True

            # ZY shapes layer (Reg ROI): display (Z, Y), row=Z, col=Y
            Z, Y_zy = zy_proj_a.shape
            zy_rect = [[(0, 0), (0, Y_zy - 1), (Z - 1, Y_zy - 1), (Z - 1, 0)]]
            self.shapes_zy_reg_layer = self.viewer.add_shapes(
                zy_rect,
                shape_type="rectangle",
                name="Reg. ZY Crop ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_zy_a_layer.translate,
            )
            self.shapes_zy_reg_layer.editable = True

            # XZ shapes layer (Reg ROI): display (X, Z), row=X, col=Z
            X_xz, Z_xz = xz_proj_a_t.shape
            xz_rect = [[(0, 0), (0, Z_xz - 1), (X_xz - 1, Z_xz - 1), (X_xz - 1, 0)]]
            self.shapes_xz_reg_layer = self.viewer.add_shapes(
                xz_rect,
                shape_type="rectangle",
                name="Reg. XZ Crop ROI",
                edge_color="orange",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_xz_a_layer.translate,
            )
            self.shapes_xz_reg_layer.editable = True

            # YX shapes layer (Apply ROI): display (X, Y), row=X, col=Y
            self.shapes_yx_layer = self.viewer.add_shapes(
                yx_rect,
                shape_type="rectangle",
                name="YX Crop ROI",
                edge_color="lime",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_yx_a_layer.translate,
            )
            self.shapes_yx_layer.editable = True

            # ZY shapes layer (Apply ROI): display (Z, Y), row=Z, col=Y
            self.shapes_zy_layer = self.viewer.add_shapes(
                zy_rect,
                shape_type="rectangle",
                name="ZY Crop ROI",
                edge_color="lime",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_zy_a_layer.translate,
            )
            self.shapes_zy_layer.editable = True

            # XZ shapes layer (Apply ROI): display (X, Z), row=X, col=Z
            self.shapes_xz_layer = self.viewer.add_shapes(
                xz_rect,
                shape_type="rectangle",
                name="XZ Crop ROI",
                edge_color="lime",
                face_color="transparent",
                edge_width=2,
                scale=(pixel_size, pixel_size),
                translate=self.projection_xz_a_layer.translate,
            )
            self.shapes_xz_layer.editable = True

            # Connect shapes data change events for cross-view sync (Apply layers)
            self.shapes_yx_layer.events.data.connect(self.on_yx_shapes_changed)
            self.shapes_zy_layer.events.data.connect(self.on_zy_shapes_changed)
            self.shapes_xz_layer.events.data.connect(self.on_xz_shapes_changed)

            # Connect shapes data change events for cross-view sync (Reg layers)
            self.shapes_yx_reg_layer.events.data.connect(self.on_yx_reg_shapes_changed)
            self.shapes_zy_reg_layer.events.data.connect(self.on_zy_reg_shapes_changed)
            self.shapes_xz_reg_layer.events.data.connect(self.on_xz_reg_shapes_changed)

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

    # ---------------------------------------------------------------------
    # Crop shapes layer helpers
    # ---------------------------------------------------------------------

    def _commit_shape_edits(self, layer):
        """Force a shapes layer to commit any pending interactive edits."""
        if layer.mode in ("transform", "direct"):
            previous_mode = layer.mode
            layer.mode = "pan_zoom"
            layer.mode = previous_mode

    def on_yx_shapes_changed(self):
        """Handle changes to YX shapes layer - sync Y to ZY, X to XZ.

        YX display (X, Y): row=X, col=Y
        ZY display (Z, Y): row=Z, col=Y  -> sync Y (col)
        XZ display (X, Z): row=X, col=Z  -> sync X (row)
        """
        if self._syncing_shapes:
            return
        self._syncing_shapes = True
        try:
            if not self.shapes_yx_layer or len(self.shapes_yx_layer.data) == 0:
                return
            yx_corners = list(self.shapes_yx_layer.data[0])
            y_min = int(min(c[1] for c in yx_corners))
            y_max = int(max(c[1] for c in yx_corners))
            x_min = int(min(c[0] for c in yx_corners))
            x_max = int(max(c[0] for c in yx_corners))

            # Sync Y to ZY layer (col = Y)
            if self.shapes_zy_layer and len(self.shapes_zy_layer.data) > 0:
                zy_corners = list(self.shapes_zy_layer.data[0])
                zy_y_min = min(c[1] for c in zy_corners)
                zy_y_max = max(c[1] for c in zy_corners)
                new_zy_corners = []
                for corner in zy_corners:
                    z, y = corner
                    new_y = y_min if y == zy_y_min else y_max
                    new_zy_corners.append([z, new_y])
                self.shapes_zy_layer.data = [new_zy_corners]

            # Sync X to XZ layer (row = X)
            if self.shapes_xz_layer and len(self.shapes_xz_layer.data) > 0:
                xz_corners = list(self.shapes_xz_layer.data[0])
                xz_x_min = min(c[0] for c in xz_corners)
                xz_x_max = max(c[0] for c in xz_corners)
                new_xz_corners = []
                for corner in xz_corners:
                    x, z = corner
                    new_x = x_min if x == xz_x_min else x_max
                    new_xz_corners.append([new_x, z])
                self.shapes_xz_layer.data = [new_xz_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_shapes = False

    def on_zy_shapes_changed(self):
        """Handle changes to ZY shapes layer - sync Y to YX, Z to XZ.

        ZY display (Z, Y): row=Z, col=Y
        YX display (X, Y): row=X, col=Y  -> sync Y (col)
        XZ display (X, Z): row=X, col=Z  -> sync Z (col)
        """
        if self._syncing_shapes:
            return
        self._syncing_shapes = True
        try:
            if not self.shapes_zy_layer or len(self.shapes_zy_layer.data) == 0:
                return
            zy_corners = list(self.shapes_zy_layer.data[0])
            y_min = int(min(c[1] for c in zy_corners))
            y_max = int(max(c[1] for c in zy_corners))
            z_min = int(min(c[0] for c in zy_corners))
            z_max = int(max(c[0] for c in zy_corners))

            # Sync Y to YX layer (col = Y)
            if self.shapes_yx_layer and len(self.shapes_yx_layer.data) > 0:
                yx_corners = list(self.shapes_yx_layer.data[0])
                yx_y_min = min(c[1] for c in yx_corners)
                yx_y_max = max(c[1] for c in yx_corners)
                new_yx_corners = []
                for corner in yx_corners:
                    x, y = corner
                    new_y = y_min if y == yx_y_min else y_max
                    new_yx_corners.append([x, new_y])
                self.shapes_yx_layer.data = [new_yx_corners]

            # Sync Z to XZ layer (col = Z)
            if self.shapes_xz_layer and len(self.shapes_xz_layer.data) > 0:
                xz_corners = list(self.shapes_xz_layer.data[0])
                xz_z_min = min(c[1] for c in xz_corners)
                xz_z_max = max(c[1] for c in xz_corners)
                new_xz_corners = []
                for corner in xz_corners:
                    x, z = corner
                    new_z = z_min if z == xz_z_min else z_max
                    new_xz_corners.append([x, new_z])
                self.shapes_xz_layer.data = [new_xz_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_shapes = False

    def on_xz_shapes_changed(self):
        """Handle changes to XZ shapes layer - sync X to YX, Z to ZY.

        XZ display (X, Z): row=X, col=Z
        YX display (X, Y): row=X, col=Y  -> sync X (row)
        ZY display (Z, Y): row=Z, col=Y  -> sync Z (row)
        """
        if self._syncing_shapes:
            return
        self._syncing_shapes = True
        try:
            if not self.shapes_xz_layer or len(self.shapes_xz_layer.data) == 0:
                return
            xz_corners = list(self.shapes_xz_layer.data[0])
            x_min = int(min(c[0] for c in xz_corners))
            x_max = int(max(c[0] for c in xz_corners))
            z_min = int(min(c[1] for c in xz_corners))
            z_max = int(max(c[1] for c in xz_corners))

            # Sync X to YX layer (row = X)
            if self.shapes_yx_layer and len(self.shapes_yx_layer.data) > 0:
                yx_corners = list(self.shapes_yx_layer.data[0])
                yx_x_min = min(c[0] for c in yx_corners)
                yx_x_max = max(c[0] for c in yx_corners)
                new_yx_corners = []
                for corner in yx_corners:
                    x, y = corner
                    new_x = x_min if x == yx_x_min else x_max
                    new_yx_corners.append([new_x, y])
                self.shapes_yx_layer.data = [new_yx_corners]

            # Sync Z to ZY layer (row = Z)
            if self.shapes_zy_layer and len(self.shapes_zy_layer.data) > 0:
                zy_corners = list(self.shapes_zy_layer.data[0])
                zy_z_min = min(c[0] for c in zy_corners)
                zy_z_max = max(c[0] for c in zy_corners)
                new_zy_corners = []
                for corner in zy_corners:
                    z, y = corner
                    new_z = z_min if z == zy_z_min else z_max
                    new_zy_corners.append([new_z, y])
                self.shapes_zy_layer.data = [new_zy_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_shapes = False

    def on_yx_reg_shapes_changed(self):
        """Handle changes to Reg YX shapes layer - sync Y to Reg ZY, X to Reg XZ.

        YX display (X, Y): row=X, col=Y
        ZY display (Z, Y): row=Z, col=Y  -> sync Y (col)
        XZ display (X, Z): row=X, col=Z  -> sync X (row)
        """
        if self._syncing_reg_shapes:
            return
        self._syncing_reg_shapes = True
        try:
            if not self.shapes_yx_reg_layer or len(self.shapes_yx_reg_layer.data) == 0:
                return
            yx_corners = list(self.shapes_yx_reg_layer.data[0])
            y_min = int(min(c[1] for c in yx_corners))
            y_max = int(max(c[1] for c in yx_corners))
            x_min = int(min(c[0] for c in yx_corners))
            x_max = int(max(c[0] for c in yx_corners))

            # Sync Y to Reg ZY layer (col = Y)
            if self.shapes_zy_reg_layer and len(self.shapes_zy_reg_layer.data) > 0:
                zy_corners = list(self.shapes_zy_reg_layer.data[0])
                zy_y_min = min(c[1] for c in zy_corners)
                zy_y_max = max(c[1] for c in zy_corners)
                new_zy_corners = []
                for corner in zy_corners:
                    z, y = corner
                    new_y = y_min if y == zy_y_min else y_max
                    new_zy_corners.append([z, new_y])
                self.shapes_zy_reg_layer.data = [new_zy_corners]

            # Sync X to Reg XZ layer (row = X)
            if self.shapes_xz_reg_layer and len(self.shapes_xz_reg_layer.data) > 0:
                xz_corners = list(self.shapes_xz_reg_layer.data[0])
                xz_x_min = min(c[0] for c in xz_corners)
                xz_x_max = max(c[0] for c in xz_corners)
                new_xz_corners = []
                for corner in xz_corners:
                    x, z = corner
                    new_x = x_min if x == xz_x_min else x_max
                    new_xz_corners.append([new_x, z])
                self.shapes_xz_reg_layer.data = [new_xz_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_reg_shapes = False

    def on_zy_reg_shapes_changed(self):
        """Handle changes to Reg ZY shapes layer - sync Y to Reg YX, Z to Reg XZ.

        ZY display (Z, Y): row=Z, col=Y
        YX display (X, Y): row=X, col=Y  -> sync Y (col)
        XZ display (X, Z): row=X, col=Z  -> sync Z (col)
        """
        if self._syncing_reg_shapes:
            return
        self._syncing_reg_shapes = True
        try:
            if not self.shapes_zy_reg_layer or len(self.shapes_zy_reg_layer.data) == 0:
                return
            zy_corners = list(self.shapes_zy_reg_layer.data[0])
            y_min = int(min(c[1] for c in zy_corners))
            y_max = int(max(c[1] for c in zy_corners))
            z_min = int(min(c[0] for c in zy_corners))
            z_max = int(max(c[0] for c in zy_corners))

            # Sync Y to Reg YX layer (col = Y)
            if self.shapes_yx_reg_layer and len(self.shapes_yx_reg_layer.data) > 0:
                yx_corners = list(self.shapes_yx_reg_layer.data[0])
                yx_y_min = min(c[1] for c in yx_corners)
                yx_y_max = max(c[1] for c in yx_corners)
                new_yx_corners = []
                for corner in yx_corners:
                    x, y = corner
                    new_y = y_min if y == yx_y_min else y_max
                    new_yx_corners.append([x, new_y])
                self.shapes_yx_reg_layer.data = [new_yx_corners]

            # Sync Z to Reg XZ layer (col = Z)
            if self.shapes_xz_reg_layer and len(self.shapes_xz_reg_layer.data) > 0:
                xz_corners = list(self.shapes_xz_reg_layer.data[0])
                xz_z_min = min(c[1] for c in xz_corners)
                xz_z_max = max(c[1] for c in xz_corners)
                new_xz_corners = []
                for corner in xz_corners:
                    x, z = corner
                    new_z = z_min if z == xz_z_min else z_max
                    new_xz_corners.append([x, new_z])
                self.shapes_xz_reg_layer.data = [new_xz_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_reg_shapes = False

    def on_xz_reg_shapes_changed(self):
        """Handle changes to Reg XZ shapes layer - sync X to Reg YX, Z to Reg ZY.

        XZ display (X, Z): row=X, col=Z
        YX display (X, Y): row=X, col=Y  -> sync X (row)
        ZY display (Z, Y): row=Z, col=Y  -> sync Z (row)
        """
        if self._syncing_reg_shapes:
            return
        self._syncing_reg_shapes = True
        try:
            if not self.shapes_xz_reg_layer or len(self.shapes_xz_reg_layer.data) == 0:
                return
            xz_corners = list(self.shapes_xz_reg_layer.data[0])
            x_min = int(min(c[0] for c in xz_corners))
            x_max = int(max(c[0] for c in xz_corners))
            z_min = int(min(c[1] for c in xz_corners))
            z_max = int(max(c[1] for c in xz_corners))

            # Sync X to Reg YX layer (row = X)
            if self.shapes_yx_reg_layer and len(self.shapes_yx_reg_layer.data) > 0:
                yx_corners = list(self.shapes_yx_reg_layer.data[0])
                yx_x_min = min(c[0] for c in yx_corners)
                yx_x_max = max(c[0] for c in yx_corners)
                new_yx_corners = []
                for corner in yx_corners:
                    x, y = corner
                    new_x = x_min if x == yx_x_min else x_max
                    new_yx_corners.append([new_x, y])
                self.shapes_yx_reg_layer.data = [new_yx_corners]

            # Sync Z to Reg ZY layer (row = Z)
            if self.shapes_zy_reg_layer and len(self.shapes_zy_reg_layer.data) > 0:
                zy_corners = list(self.shapes_zy_reg_layer.data[0])
                zy_z_min = min(c[0] for c in zy_corners)
                zy_z_max = max(c[0] for c in zy_corners)
                new_zy_corners = []
                for corner in zy_corners:
                    z, y = corner
                    new_z = z_min if z == zy_z_min else z_max
                    new_zy_corners.append([new_z, y])
                self.shapes_zy_reg_layer.data = [new_zy_corners]
        finally:
            self._clamp_reg_to_apply()
            self._syncing_reg_shapes = False

    def _clamp_reg_to_apply(self):
        """Clamp Reg ROI layers to fit within Apply ROI bounds.

        After the user modifies either set of ROIs, this ensures the Reg ROI
        is always within the Apply ROI. If Apply shrinks, Reg is clamped.
        If Apply grows, Reg stays where it is (unless already outside).
        """
        # Guard against layers not yet created
        if (self.shapes_yx_layer is None or self.shapes_yx_reg_layer is None or
                self.shapes_zy_layer is None or self.shapes_zy_reg_layer is None or
                self.shapes_xz_layer is None or self.shapes_xz_reg_layer is None):
            return
        if (len(self.shapes_yx_layer.data) == 0 or
                len(self.shapes_yx_reg_layer.data) == 0):
            return

        # Prevent recursive updates in both sync systems
        self._syncing_reg_shapes = True
        try:
            # Extract Apply bounds
            apply_yx = list(self.shapes_yx_layer.data[0])
            apply_y_min = int(min(c[1] for c in apply_yx))
            apply_y_max = int(max(c[1] for c in apply_yx))
            apply_x_min = int(min(c[0] for c in apply_yx))
            apply_x_max = int(max(c[0] for c in apply_yx))

            apply_zy = list(self.shapes_zy_layer.data[0])
            apply_z_min = int(min(c[0] for c in apply_zy))
            apply_z_max = int(max(c[0] for c in apply_zy))

            changed = False

            # Clamp Reg YX layer
            reg_yx = list(self.shapes_yx_reg_layer.data[0])
            reg_y_min = int(min(c[1] for c in reg_yx))
            reg_y_max = int(max(c[1] for c in reg_yx))
            reg_x_min = int(min(c[0] for c in reg_yx))
            reg_x_max = int(max(c[0] for c in reg_yx))

            new_reg_y_min = max(reg_y_min, apply_y_min)
            new_reg_y_max = min(reg_y_max, apply_y_max)
            new_reg_x_min = max(reg_x_min, apply_x_min)
            new_reg_x_max = min(reg_x_max, apply_x_max)

            if (new_reg_y_min != reg_y_min or new_reg_y_max != reg_y_max or
                    new_reg_x_min != reg_x_min or new_reg_x_max != reg_x_max):
                new_corners = []
                for corner in reg_yx:
                    x, y = corner
                    new_x = new_reg_x_min if x == reg_x_min else new_reg_x_max
                    new_y = new_reg_y_min if y == reg_y_min else new_reg_y_max
                    new_corners.append([new_x, new_y])
                self.shapes_yx_reg_layer.data = [new_corners]
                changed = True

            # Clamp Reg ZY layer
            reg_zy = list(self.shapes_zy_reg_layer.data[0])
            reg_zy_y_min = int(min(c[1] for c in reg_zy))
            reg_zy_y_max = int(max(c[1] for c in reg_zy))
            reg_z_min = int(min(c[0] for c in reg_zy))
            reg_z_max = int(max(c[0] for c in reg_zy))

            new_reg_zy_y_min = max(reg_zy_y_min, apply_y_min)
            new_reg_zy_y_max = min(reg_zy_y_max, apply_y_max)
            new_reg_z_min = max(reg_z_min, apply_z_min)
            new_reg_z_max = min(reg_z_max, apply_z_max)

            if (new_reg_zy_y_min != reg_zy_y_min or new_reg_zy_y_max != reg_zy_y_max or
                    new_reg_z_min != reg_z_min or new_reg_z_max != reg_z_max):
                new_corners = []
                for corner in reg_zy:
                    z, y = corner
                    new_z = new_reg_z_min if z == reg_z_min else new_reg_z_max
                    new_y = new_reg_zy_y_min if y == reg_zy_y_min else new_reg_zy_y_max
                    new_corners.append([new_z, new_y])
                self.shapes_zy_reg_layer.data = [new_corners]
                changed = True

            # Clamp Reg XZ layer
            reg_xz = list(self.shapes_xz_reg_layer.data[0])
            reg_xz_x_min = int(min(c[0] for c in reg_xz))
            reg_xz_x_max = int(max(c[0] for c in reg_xz))
            reg_xz_z_min = int(min(c[1] for c in reg_xz))
            reg_xz_z_max = int(max(c[1] for c in reg_xz))

            new_reg_xz_x_min = max(reg_xz_x_min, apply_x_min)
            new_reg_xz_x_max = min(reg_xz_x_max, apply_x_max)
            new_reg_xz_z_min = max(reg_xz_z_min, apply_z_min)
            new_reg_xz_z_max = min(reg_xz_z_max, apply_z_max)

            if (new_reg_xz_x_min != reg_xz_x_min or new_reg_xz_x_max != reg_xz_x_max or
                    new_reg_xz_z_min != reg_xz_z_min or new_reg_xz_z_max != reg_xz_z_max):
                new_corners = []
                for corner in reg_xz:
                    x, z = corner
                    new_x = new_reg_xz_x_min if x == reg_xz_x_min else new_reg_xz_x_max
                    new_z = new_reg_xz_z_min if z == reg_xz_z_min else new_reg_xz_z_max
                    new_corners.append([new_x, new_z])
                self.shapes_xz_reg_layer.data = [new_corners]
                changed = True
        finally:
            self._syncing_reg_shapes = False

    def _extract_crop_bounds(self, for_registration: bool = False) -> tuple:
        """Extract 3D crop bounds from shapes layers.

        Parameters
        ----------
        for_registration : bool
            If True, extract from Reg layers. If False, extract from Apply layers.

        Returns
        -------
        tuple
            (z_start, z_end, y_start, y_end, x_start, x_end) in pixel units
            relative to deskewed View A coordinates.
        """
        if for_registration:
            yx_layer = self.shapes_yx_reg_layer
            zy_layer = self.shapes_zy_reg_layer
            xz_layer = self.shapes_xz_reg_layer
        else:
            yx_layer = self.shapes_yx_layer
            zy_layer = self.shapes_zy_layer
            xz_layer = self.shapes_xz_layer

        # Commit pending interactive edits
        self._commit_shape_edits(yx_layer)
        self._commit_shape_edits(zy_layer)
        self._commit_shape_edits(xz_layer)

        # Extract from YX (display: X, Y) -> row=X, col=Y
        yx_corners = list(yx_layer.data[0])
        x_start = int(min(c[0] for c in yx_corners))
        x_end = int(max(c[0] for c in yx_corners)) + 1
        y_start = int(min(c[1] for c in yx_corners))
        y_end = int(max(c[1] for c in yx_corners)) + 1

        # Extract from ZY (display: Z, Y) -> row=Z, col=Y
        zy_corners = list(zy_layer.data[0])
        z_start = int(min(c[0] for c in zy_corners))
        z_end = int(max(c[0] for c in zy_corners)) + 1

        return (z_start, z_end, y_start, y_end, x_start, x_end)

    def _format_transform_details(self, transform_matrix, transform_type):
        """Format the fitted transform components as an HTML string for display.

        Uses ``decompose_affine_matrix`` to extract translation, rotation, scale,
        and shear from the fitted registration matrix, then shows only the
        components that are relevant for the selected transform type.

        Args:
            transform_matrix: 4x4 affine transformation matrix (NumPy array).
            transform_type: One of "t", "t+r", "t+sh", "t+ssh", "t+r+s", "t+sh+s".

        Returns:
            HTML-formatted string with the relevant transform components.
        """
        try:
            comp = decompose_affine_matrix(np.asarray(transform_matrix, dtype=float))
        except (ValueError, np.linalg.LinAlgError):
            return '<br><i>Could not decompose transform matrix.</i>'

        lines = []

        # Translation is always shown (in pixels)
        tx, ty, tz = comp.translation
        lines.append(
            f'Translation (px): tx={tx:.2f}, ty={ty:.2f}, tz={tz:.2f}'
        )

        has_rotation = "r" in transform_type
        has_shear = "sh" in transform_type
        has_scale = transform_type.endswith("+s")

        if has_rotation:
            alpha_deg = np.degrees(comp.euler_angles[0])
            beta_deg = np.degrees(comp.euler_angles[1])
            gamma_deg = np.degrees(comp.euler_angles[2])
            lines.append(
                f'Rotation (deg): α={alpha_deg:.2f}, β={beta_deg:.2f}, γ={gamma_deg:.2f}'
            )

        if has_shear:
            h_xy, h_xz, h_yz = comp.shear
            lines.append(
                f'Shear: h_xy={h_xy:.4f}, h_xz={h_xz:.4f}, h_yz={h_yz:.4f}'
            )

        if has_scale:
            sx, sy, sz = comp.scale
            lines.append(
                f'Scale: sx={sx:.4f}, sy={sy:.4f}, sz={sz:.4f}'
            )

        return f"<br><b>Transform Details:</b><br>" + "<br>".join(lines)

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

    def register_data(self):
        """Register the deskewed data.

        Uses remote server when connection is active and path is remote,
        otherwise falls back to local execution using RegistrationWorker.
        """
        # In remote mode, we only need a valid session ID (volumes are on server)
        # In local mode, we need the deskewed volumes locally
        if self._remote_mode:
            if not self._session_id:
                QMessageBox.warning(
                    self, "Error", "No active session. Please load and deskew data first."
                )
                return
        else:
            if self.a_deskewed is None or self.b_deskewed is None:
                QMessageBox.warning(
                    self, "Error", "Please load and deskew data first"
                )
                return

        self.register_button.setEnabled(False)
        self.load_deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.status_label.setText("Starting registration...")

        # Branch to local or remote execution
        if self._remote_mode and self.remote_client:
            self._register_remote()
        else:
            self._register_local()

    def _register_local(self):
        """Register locally using RegistrationWorker."""
        transform_type = self.transform_combo.currentText()

        # Convert pre-reg transform from micrometers to pixel units
        pixel_size = self.pixel_size_spin.value()
        t0 = [t / pixel_size for t in self._pre_reg_transform]
        tz, ty, tx = t0

        # Extract crop bounds from Reg shapes layers (for registration)
        z_start, z_end, y_start, y_end, x_start, x_end = self._extract_crop_bounds(for_registration=True)

        # Crop View A to crop region
        crop_slices_a = (
            slice(z_start, z_end),
            slice(y_start, y_end),
            slice(x_start, x_end),
        )

        # Compute corresponding crop in View B coordinates (shift by -pre-reg translation)
        Zb, Yb, Xb = self.b_deskewed.shape
        z_start_b = max(0, int(z_start - tz))
        z_end_b = min(Zb, int(z_end - tz))
        y_start_b = max(0, int(y_start - ty))
        y_end_b = min(Yb, int(y_end - ty))
        x_start_b = max(0, int(x_start - tx))
        x_end_b = min(Xb, int(x_end - tx))

        crop_slices_b = (
            slice(z_start_b, z_end_b),
            slice(y_start_b, y_end_b),
            slice(x_start_b, x_end_b),
        )

        # Crop both volumes
        a_vol = self.a_deskewed[crop_slices_a]
        b_vol = self.b_deskewed[crop_slices_b]

        # Use min size to guarantee both volumes have the same shape
        min_shape = tuple(min(a, b) for a, b in zip(a_vol.shape, b_vol.shape))
        a_vol = a_vol[:min_shape[0], :min_shape[1], :min_shape[2]]
        b_vol = b_vol[:min_shape[0], :min_shape[1], :min_shape[2]]

        # Initial translation is always [0, 0, 0] (absorbed by crop)
        initial_t = [0, 0, 0]

        self.update_progress(
            f"Cropped to region: {min_shape[0]}x{min_shape[1]}x{min_shape[2]}"
        )

        # Map metric display name to internal string
        metric_map = {
            "Norm. Inner Prod.": "nip",
            "Corr. Ratio": "cr",
            "Norm. XCorr": "ncc",
        }
        metric = metric_map.get(self.metric_combo.currentText(), "cr")

        # Map interpolation method display name to internal string
        interp_method_map = {
            "Linear": "linear",
            "Cubic Spline": "cubspl",
        }
        interp_method = interp_method_map.get(self.interp_method_combo.currentText(), "cubspl")

        use_piecewise = self.piecewise_checkbox.isChecked()

        # Read bound values from UI
        bound_translation = self.bound_translation_spin.value()
        bound_rot_shear = self.bound_rot_shear_spin.value()
        bound_scale = self.bound_scale_spin.value()

        # Create and start worker
        use_remote = (self.remote_client is not None and self.remote_client.is_connected)
        self.reg_worker = RegistrationWorker(
            a_vol, b_vol, transform_type,
            initial_translation=initial_t,
            metric=metric,
            interp_method=interp_method,
            opt_method=self.opt_method_combo.currentText(),
            use_piecewise=use_piecewise,
            bound_translation=bound_translation,
            bound_rot_shear=bound_rot_shear,
            bound_scale=bound_scale,
            use_remote=use_remote,
            remote_client=self.remote_client,
        )
        self.reg_worker.registered.connect(self.on_registered)
        self.reg_worker.error_occurred.connect(self.on_error)
        self.reg_worker.progress_updated.connect(self.update_progress)
        self.reg_worker.start()

    def _register_remote(self):
        """Register via remote server using signal-based pattern."""
        if not self.remote_client:
            self.on_error("Remote client not available")
            return

        # Convert pre-reg transform from micrometers to pixel units
        pixel_size = self.pixel_size_spin.value()
        pre_reg_translation = [t / pixel_size for t in self._pre_reg_transform]

        # Map metric display name to internal string
        metric_map = {
            "Norm. Inner Prod.": "nip",
            "Corr. Ratio": "cr",
            "Norm. XCorr": "ncc",
        }
        metric = metric_map.get(self.metric_combo.currentText(), "cr")

        # Map interpolation method display name to internal string
        interp_method_map = {
            "Linear": "linear",
            "Cubic Spline": "cubspl",
        }
        interp_method = interp_method_map.get(self.interp_method_combo.currentText(), "cubspl")

        # Extract crop bounds from Reg shapes layers (for registration)
        z_start, z_end, y_start, y_end, x_start, x_end = self._extract_crop_bounds(for_registration=True)

        params = {
            "session_id": self._session_id,
            "transform_type": self.transform_combo.currentText(),
            "pre_reg_translation": pre_reg_translation,
            "crop_bounds": {
                "z_start": z_start, "z_end": z_end,
                "y_start": y_start, "y_end": y_end,
                "x_start": x_start, "x_end": x_end,
            },
            "metric": metric,
            "interp_method": interp_method,
            "use_piecewise": self.piecewise_checkbox.isChecked(),
            "bound_translation": self.bound_translation_spin.value(),
            "bound_rot_shear": self.bound_rot_shear_spin.value(),
            "bound_scale": self.bound_scale_spin.value(),
            "opt_method": self.opt_method_combo.currentText(),
        }

        # Store context for signal handler
        self._pending_command_type = "register"

        try:
            self.remote_client.send_command(
                command="register",
                params=params,
                callback=None,
                progress_callback=None,
            )
        except Exception as e:
            self.on_error(f"Failed to send command: {e}")

    def _add_registered_projection_layers(
        self, yx_proj_b, zy_proj_b, xz_proj_b,
        yx_proj_a=None, zy_proj_a=None, xz_proj_a=None
    ):
        """Add max projection layers for registered data.

        Adds B-registered projections in yellow, and optionally cropped View A
        projections in magenta. Hides the original pre-registration projection
        layers so the user can focus on the registration result.

        Args:
            yx_proj_b: YX projection of registered B, shape (Y, X)
            zy_proj_b: ZY projection of registered B, shape (Z, Y)
            xz_proj_b: XZ projection of registered B, shape (Z, X)
            yx_proj_a: Optional YX projection of cropped A, shape (Y, X)
            zy_proj_a: Optional ZY projection of cropped A, shape (Z, Y)
            xz_proj_a: Optional XZ projection of cropped A, shape (Z, X)
        """
        pixel_size = self._pixel_size
        if pixel_size is None:
            return

        # Hide original pre-registration projection layers
        for layer in [
            self.projection_yx_a_layer, self.projection_yx_b_layer,
            self.projection_zy_a_layer, self.projection_zy_b_layer,
            self.projection_xz_a_layer, self.projection_xz_b_layer,
        ]:
            if layer is not None:
                layer.visible = False

        # Transpose B projections for display (matching existing pattern)
        yx_b_t = yx_proj_b.T  # (X, Y)
        xz_b_t = xz_proj_b.T  # (X, Z)

        # Calculate offsets matching existing projection layout
        Z_b, Y_zy_b = zy_proj_b.shape
        offset_z_um = -Z_b * pixel_size
        offset_x_um = Y_zy_b * pixel_size

        # Remove existing registered projection layers if any
        for layer_attr in [
            "projection_yx_b_reg_layer", "projection_zy_b_reg_layer",
            "projection_xz_b_reg_layer",
            "projection_yx_a_reg_layer", "projection_zy_a_reg_layer",
            "projection_xz_a_reg_layer",
        ]:
            layer = getattr(self, layer_attr)
            if layer is not None:
                try:
                    self.viewer.layers.remove(layer)
                except (ValueError, TypeError):
                    pass

        # Add YX projection for B Registered — overlays View A YX position
        self.projection_yx_b_reg_layer = self.viewer.add_image(
            yx_b_t, name="YX Projection (B Registered)",
            colormap="yellow", opacity=0.9, blending="additive",
            scale=(pixel_size, pixel_size),
        )

        # Add ZY projection for B Registered — overlays View A ZY position
        self.projection_zy_b_reg_layer = self.viewer.add_image(
            zy_proj_b, name="ZY Projection (B Registered)",
            colormap="yellow", opacity=0.9, blending="additive",
            scale=(pixel_size, pixel_size),
            translate=(offset_z_um, 0),
        )

        # Add XZ projection for B Registered — overlays View A XZ position
        self.projection_xz_b_reg_layer = self.viewer.add_image(
            xz_b_t, name="XZ Projection (B Registered)",
            colormap="yellow", opacity=0.9, blending="additive",
            scale=(pixel_size, pixel_size),
            translate=(0, offset_x_um),
        )

        # Add cropped View A layers if projections provided
        if yx_proj_a is not None and zy_proj_a is not None and xz_proj_a is not None:
            yx_a_t = yx_proj_a.T  # (X, Y)
            xz_a_t = xz_proj_a.T  # (X, Z)

            Z_a, Y_zy_a = zy_proj_a.shape
            offset_z_um_a = -Z_a * pixel_size
            offset_x_um_a = Y_zy_a * pixel_size

            # YX
            self.projection_yx_a_reg_layer = self.viewer.add_image(
                yx_a_t, name="YX Projection (A Cropped)",
                colormap="magenta", opacity=0.9, blending="additive",
                scale=(pixel_size, pixel_size),
            )

            # ZY
            self.projection_zy_a_reg_layer = self.viewer.add_image(
                zy_proj_a, name="ZY Projection (A Cropped)",
                colormap="magenta", opacity=0.9, blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(offset_z_um_a, 0),
            )

            # XZ
            self.projection_xz_a_reg_layer = self.viewer.add_image(
                xz_a_t, name="XZ Projection (A Cropped)",
                colormap="magenta", opacity=0.9, blending="additive",
                scale=(pixel_size, pixel_size),
                translate=(0, offset_x_um_a),
            )

    def on_registered(self, result):
        """Handle successful registration."""
        a_registered = result["a_registered"]
        b_registered = result["b_registered"]
        transform_matrix = result["transform_matrix"]
        cr = result["correlation_ratio"]
        # Store for saving
        self.registration_matrix = transform_matrix
        self.correlation_ratio = cr

        # Compute max projections from registered B volume
        yx_proj_b = np.max(b_registered, axis=0)  # (Y, X)
        zy_proj_b = np.max(b_registered, axis=2)  # (Z, Y)
        xz_proj_b = np.max(b_registered, axis=1)  # (Z, X)

        # Always compute A projections (cropped volume is always returned)
        yx_proj_a = np.max(a_registered, axis=0)  # (Y, X)
        zy_proj_a = np.max(a_registered, axis=2)  # (Z, Y)
        xz_proj_a = np.max(a_registered, axis=1)  # (Z, X)

        # Add registered projection layers
        self._add_registered_projection_layers(
            yx_proj_b, zy_proj_b, xz_proj_b,
            yx_proj_a, zy_proj_a, xz_proj_a
        )

        # Update status and results
        self.status_label.setText(
            f"Registration completed! A: {a_registered.shape}, B: {b_registered.shape}"
        )

        transform_details = self._format_transform_details(
            transform_matrix, result["transform_type"]
        )
        results_text = f"""
        <b>Registration Results:</b><br>
        Metric: {cr:.3f}<br>
        Shape: {a_registered.shape}<br>
        {transform_details}
        """
        self.results_label.setText(results_text)

        self.register_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        self.save_button.setEnabled(True)
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
            "theta": math.pi / 4,  # Hardcoded to 45 degrees
        }
        self.registered.emit(output_data)

    def _on_remote_command_response(self, response: dict):
        """Handle command_response signal from RemoteClient (runs on main thread).

        Only handles responses for commands we sent (load_deskew, query_positions).
        """
        if self._pending_command_type == "load_deskew":
            self._pending_command_type = None
            if response.get("success"):
                result = response.get("result", {})
                # Convert volume_shape lists back to tuples for consistency
                if "volume_shape_a" in result and isinstance(result["volume_shape_a"], list):
                    result["volume_shape_a"] = tuple(result["volume_shape_a"])
                if "volume_shape_b" in result and isinstance(result["volume_shape_b"], list):
                    result["volume_shape_b"] = tuple(result["volume_shape_b"])
                # Store session ID for later registration
                self._session_id = result.get("session_id")
                self._pending_data_path = None
                self.on_load_deskew_ready(result)
            else:
                error_msg = response.get("error", "Unknown error")
                self._pending_data_path = None
                self.on_error(error_msg)
        elif self._pending_command_type == "query_positions":
            self._pending_command_type = None
            if response.get("success"):
                num_pos = response.get("result", {}).get("num_positions", 1)
                self.position_spin.setRange(0, max(0, num_pos - 1))
        elif self._pending_command_type == "register":
            self._pending_command_type = None
            if response.get("success"):
                result = response.get("result", {})
                self._on_register_ready_main(result)
            else:
                error_msg = response.get("error", "Unknown error")
                self.on_error(error_msg)
        elif self._pending_command_type == "save_params":
            self._pending_command_type = None
            if response.get("success"):
                self._remote_params_saved = True
                self.status_label.setText("Registration parameters saved on remote server!")
                self._update_apply_section()
            else:
                error_msg = response.get("error", "Unknown error")
                QMessageBox.critical(self, "Error", f"Failed to save parameters: {error_msg}")
        elif self._pending_command_type == "apply_registration":
            self._pending_command_type = None
            if response.get("success"):
                result = response.get("result", {})
                output_folder = result.get("output_folder", "unknown")
                self.status_label.setText(f"Apply completed! Output: {output_folder}")
                self.progress_bar.setVisible(False)
                self.apply_button.setEnabled(True)
                self.load_deskew_button.setEnabled(True)
                # Emit registered signal with output paths for Deconvolution
                time_range = self._apply_time_range if hasattr(self, '_apply_time_range') else (0, 0)
                output_data = {
                    "a_path": os.path.join(output_folder, f"a_t{time_range[0]}.zarr"),
                    "b_path": os.path.join(output_folder, f"b_t{time_range[0]}.zarr"),
                    "output_folder": output_folder,
                }
                self.registered.emit(output_data)
            else:
                error_msg = response.get("error", "Unknown error")
                self._on_apply_error(error_msg)

    def _on_register_ready_main(self, result):
        """Handle successful remote registration results on main thread.

        The remote server returns the transform matrix, correlation ratio,
        and maximum projections of the B-registered volume.
        """
        transform_matrix = result.get("transform_matrix")
        cr = result.get("correlation_ratio", 0.0)

        # Store for saving
        self.registration_matrix = transform_matrix
        self.correlation_ratio = cr

        # Add registered projection layers from server
        # Always show cropped A projections (server always returns them now)
        yx_proj_a = result.get("yx_proj_a")
        zy_proj_a = result.get("zy_proj_a")
        xz_proj_a = result.get("xz_proj_a")

        self._add_registered_projection_layers(
            result["yx_proj_b"],
            result["zy_proj_b"],
            result["xz_proj_b"],
            yx_proj_a, zy_proj_a, xz_proj_a,
        )

        # Update status and results
        self.status_label.setText("Registration completed on remote server!")

        transform_details = self._format_transform_details(
            transform_matrix, self.transform_combo.currentText()
        )
        results_text = f"""
        <b>Registration Results:</b><br>
        Metric: {cr:.3f}<br>
        (Volumes remain on remote server)<br>
        {transform_details}
        """
        self.results_label.setText(results_text)

        self.register_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        # Emit signal with registration results (no volumes in remote mode)
        output_data = {
            "transform_matrix": transform_matrix,
            "correlation_ratio": cr,
            "transform_type": self.transform_combo.currentText(),
            "step_size": self.step_size_spin.value(),
            "pixel_size": self.pixel_size_spin.value(),
            "theta": math.pi / 4,  # Hardcoded to 45 degrees
        }
        self.registered.emit(output_data)

    def _on_remote_progress(self, message: str, percentage: int):
        """Handle progress signal from RemoteClient (runs on main thread)."""
        self.update_progress(message)
        if percentage >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)

    def on_error(self, error_msg):
        """Handle error."""
        QMessageBox.critical(self, "Error", f"Operation failed: {error_msg}")
        self.status_label.setText("Error during operation")
        self.load_deskew_button.setEnabled(True)
        self.register_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    def save_registration_params(self):
        """Save registration parameters to a JSON file in the Data Path folder.

        Uses remote server when connection is active and path is remote,
        otherwise saves locally.
        """
        data_path = self.path_edit.text()
        if not data_path:
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

        # For local mode, validate path exists
        if not self._remote_mode and not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

        # Check we have registration results
        if self.registration_matrix is None or self.correlation_ratio is None:
            QMessageBox.warning(
                self, "Error", "Please register data before saving parameters."
            )
            return

        # Build parameters dictionary
        params_dict = self._build_registration_params()

        # Branch to local or remote save
        if self._remote_mode and self.remote_client:
            self._save_params_remote(data_path, params_dict)
        else:
            self._save_params_local(data_path, params_dict)

    def _build_registration_params(self) -> dict:
        """Build the registration parameters dictionary from current UI state."""
        # Mapping dicts for display name to internal string
        metric_map = {
            "Norm. Inner Prod.": "nip",
            "Corr. Ratio": "cr",
            "Norm. XCorr": "ncc",
        }
        interp_method_map = {
            "Linear": "linear",
            "Cubic Spline": "cubspl",
        }

        # Extract Apply crop bounds (for output cropping in Apply)
        (apply_z_start, apply_z_end, apply_y_start, apply_y_end,
         apply_x_start, apply_x_end) = self._extract_crop_bounds(for_registration=False)

        # Extract Reg crop bounds (used during registration)
        (reg_z_start, reg_z_end, reg_y_start, reg_y_end,
         reg_x_start, reg_x_end) = self._extract_crop_bounds(for_registration=True)

        deskew_params = {
            "method": self.method_combo.currentText(),
            "step_size_um": self.step_size_spin.value(),
            "pixel_size_um": self.pixel_size_spin.value(),
            "theta_degrees": 45.0,  # Hardcoded to 45 degrees
            "auto_crop": self.auto_crop_checkbox.isChecked(),
            "camera_offset": self.camera_offset_spin.value(),
        }
        if self.method_combo.currentText() == "orthopsf":
            deskew_params.update({
                "psf_fwhm_axial": self.psf_fwhm_axial_spin.value(),
                "psf_fwhm_lateral": self.psf_fwhm_lateral_spin.value(),
                "psf_model": self._map_psf_model(self.psf_model_combo.currentText()),
                "use_psf_in_plane": self.use_psf_in_plane_checkbox.isChecked(),
            })

        return {
            "deskewing_parameters": deskew_params,
            "pre_reg_transform": {
                "tz_um": self._pre_reg_transform[0],
                "ty_um": self._pre_reg_transform[1],
                "tx_um": self._pre_reg_transform[2],
            },
            "registration_parameters": {
                "transform_type": self.transform_combo.currentText(),
                "metric": metric_map.get(self.metric_combo.currentText(), "cr"),
                "interp_method": interp_method_map.get(
                    self.interp_method_combo.currentText(), "cubspl"
                ),
                "use_piecewise": self.piecewise_checkbox.isChecked(),
                "bound_translation": self.bound_translation_spin.value(),
                "bound_rot_shear": self.bound_rot_shear_spin.value(),
                "bound_scale": self.bound_scale_spin.value(),
                "opt_method": self.opt_method_combo.currentText(),
            },
            "affine_registration_matrix": self.registration_matrix.tolist() if hasattr(self.registration_matrix, "tolist") else self.registration_matrix,
            "correlation_ratio": self.correlation_ratio,
            "crop_bounds": {
                "z_start": apply_z_start,
                "z_end": apply_z_end,
                "y_start": apply_y_start,
                "y_end": apply_y_end,
                "x_start": apply_x_start,
                "x_end": apply_x_end,
            },
            "reg_crop_bounds": {
                "z_start": reg_z_start,
                "z_end": reg_z_end,
                "y_start": reg_y_start,
                "y_end": reg_y_end,
                "x_start": reg_x_start,
                "x_end": reg_x_end,
            },
        }

    def _save_params_local(self, data_path: str, params_dict: dict):
        """Save parameters to a JSON file locally."""
        output_path = os.path.join(data_path, "deskew_registration_params.json")
        try:
            with open(output_path, "w") as f:
                json.dump(params_dict, f, indent=2)
            self.status_label.setText("Registration parameters saved!")
            # Enable Apply section after successful save
            self._update_apply_section()
        except (IOError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {e}")

    def _save_params_remote(self, data_path: str, params_dict: dict):
        """Save parameters via remote server using signal-based pattern."""
        if not self.remote_client:
            self.on_error("Remote client not available")
            return

        params = {
            "data_path": data_path,
            "params_dict": params_dict,
        }

        # Store context for signal handler
        self._pending_command_type = "save_params"

        try:
            self.remote_client.send_command(
                command="save_params",
                params=params,
                callback=None,
                progress_callback=None,
            )
        except Exception as e:
            self.on_error(f"Failed to send command: {e}")

    def _detect_dataset_dimensions(self):
        """Detect the number of timepoints and channels from the dataset.

        Returns:
            Tuple of (n_time, n_channel) where n_channel is the number of
            logical channels (A/B pairs), not raw TIFF channel count.
            Returns (1, 1) if detection fails.
        """
        data_path = self.path_edit.text()
        use_remote = (self.remote_client is not None and self.remote_client.is_connected)
        if not data_path or (not use_remote and not os.path.exists(data_path)):
            return (1, 1)

        multi_pos = self.multi_pos_checkbox.isChecked()
        try:
            from pyspim.data import dispim as data
            with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
                shape = acq.shape
                if multi_pos:
                    # 4D: (C, Z, Y, X) or 5D: (P, C, Z, Y, X)
                    if len(shape) == 5:
                        n_time = shape[0]  # positions as timepoints
                        n_chan = shape[1] // 2
                    else:
                        n_time = 1
                        n_chan = shape[0] // 2
                else:
                    # 4D: (C, Z, Y, X) or 5D: (T, C, Z, Y, X)
                    if len(shape) == 5:
                        n_time = shape[0]
                        n_chan = shape[1] // 2
                    else:
                        n_time = 1
                        n_chan = shape[0] // 2
                return (max(1, n_time), max(1, n_chan))
        except Exception:
            return (1, 1)

    def _update_apply_section(self):
        """Update Apply section based on existence of deskew_registration_params.json.

        In remote mode, relies on _remote_params_saved flag since os.path.exists
        does not work for remote paths.
        """
        data_path = self.path_edit.text()
        if not data_path:
            self._set_apply_enabled(False)
            return

        # Remote mode: check via SFTP first, then fall back to flag
        if self._remote_mode:
            if self._remote_params_saved:
                self._set_apply_enabled(True)
            else:
                # Check if params file exists on remote server via SFTP
                try:
                    sftp = self.remote_client.get_sftp_client()
                    params_path = f"{data_path}/deskew_registration_params.json"
                    sftp.stat(params_path)
                    # File exists
                    self._remote_params_saved = True
                    self._set_apply_enabled(True)
                except (FileNotFoundError, AttributeError):
                    self._set_apply_enabled(False)
            return

        # Local mode: check file system
        if not os.path.exists(data_path):
            self._set_apply_enabled(False)
            return

        params_path = os.path.join(data_path, "deskew_registration_params.json")
        if use_remote or os.path.exists(params_path):
            n_time, n_chan = self._detect_dataset_dimensions()
            self.time_min_spin.setRange(0, max(0, n_time - 1))
            self.time_max_spin.setRange(0, max(0, n_time - 1))
            self.time_min_spin.setValue(0)
            self.time_max_spin.setValue(n_time - 1)
            self.channel_min_spin.setRange(0, max(0, n_chan - 1))
            self.channel_max_spin.setRange(0, max(0, n_chan - 1))
            self.channel_min_spin.setValue(0)
            self.channel_max_spin.setValue(n_chan - 1)
            self._set_apply_enabled(True)
        else:
            self._set_apply_enabled(False)

    def _set_apply_enabled(self, enabled: bool):
        """Enable or disable all controls in the Apply section."""
        self.time_min_spin.setEnabled(enabled)
        self.time_max_spin.setEnabled(enabled)
        self.channel_min_spin.setEnabled(enabled)
        self.channel_max_spin.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        # Batch checkbox is only enabled in remote mode
        if enabled and self._remote_mode:
            self.batch_job_checkbox.setEnabled(True)
        else:
            self.batch_job_checkbox.setEnabled(False)
            self.batch_job_checkbox.setChecked(False)

    def apply_registration(self):
        """Apply saved registration transformations to specified ranges.

        Uses remote server when connection is active and path is remote,
        otherwise falls back to local execution using ApplyWorker.
        When batch checkbox is checked and remote connected, submits as SLURM job.
        """
        data_path = self.path_edit.text()
        if not data_path:
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return
     
        # For local mode, validate path and params file exist
        if not self._remote_mode:
            if not os.path.exists(data_path):
                QMessageBox.warning(self, "Error", "Please select a valid data path")
                return
            params_path = os.path.join(data_path, "deskew_registration_params.json")
            if not os.path.exists(params_path):
                QMessageBox.warning(self, "Error", "Registration parameters file not found")
                return

        time_min = self.time_min_spin.value()
        time_max = self.time_max_spin.value()
        chan_min = self.channel_min_spin.value()
        chan_max = self.channel_max_spin.value()

        if time_min > time_max:
            QMessageBox.warning(self, "Error", "Time min must be <= Time max")
            return
        if chan_min > chan_max:
            QMessageBox.warning(self, "Error", "Channel min must be <= Channel max")
            return

        multi_pos = self.multi_pos_checkbox.isChecked()
        position = self.position_spin.value()
        ignore_bbox = self.ignore_bbox_checkbox.isChecked()
        save_tiffs = self.save_tiffs_checkbox.isChecked()

        # Determine output folder
        if multi_pos:
            output_folder = os.path.join(data_path, f"pos{position}_dskreg")
        else:
            output_folder = os.path.join(data_path, "dskreg")

        # Store for later signal emission
        self._apply_output_folder = output_folder
        self._apply_time_range = (time_min, time_max)

        # Check for batch mode when remote connected
        if self._remote_mode and self.batch_job_checkbox.isChecked():
            self._apply_registration_batch(
                data_path, output_folder,
                (time_min, time_max), (chan_min, chan_max),
                multi_pos, position, ignore_bbox,
                save_tiffs,
            )
            return

        # Disable UI during processing
        self.apply_button.setEnabled(False)
        self.load_deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting apply...")

        # Branch to local or remote execution
        if self._remote_mode and self.remote_client:
            self._apply_registration_remote(
                data_path, output_folder,
                (time_min, time_max), (chan_min, chan_max),
                multi_pos, position, ignore_bbox,
                save_tiffs,
            )
        else:
            params_path = os.path.join(data_path, "deskew_registration_params.json")
            self.apply_worker = ApplyWorker(
                data_path, params_path,
                (time_min, time_max), (chan_min, chan_max),
                multi_pos, position, output_folder, ignore_bbox,
                save_tiffs,
            )
            self.apply_worker.finished.connect(self._on_apply_finished)
            self.apply_worker.error_occurred.connect(self._on_apply_error)
            self.apply_worker.progress_updated.connect(self._on_apply_progress)
            self.apply_worker.start()

    def _apply_registration_remote(
        self,
        data_path: str,
        output_folder: str,
        time_range: tuple,
        channel_range: tuple,
        multi_pos: bool,
        position: int,
        ignore_bbox: bool,
        save_tiffs: bool,
    ):
        """Apply registration via remote server using signal-based pattern."""
        if not self.remote_client:
            self._on_apply_error("Remote client not available")
            return

        params = {
            "data_path": data_path,
            "output_folder": output_folder,
            "time_range": time_range,
            "channel_range": channel_range,
            "multi_pos": multi_pos,
            "position": position,
            "ignore_bbox": ignore_bbox,
            "save_tiffs": save_tiffs,
        }

        self._pending_command_type = "apply_registration"

        try:
            self.remote_client.send_command(
                command="apply_registration",
                params=params,
                callback=None,
                progress_callback=None,
            )
        except Exception as e:
            self._on_apply_error(f"Failed to send command: {e}")

    def _on_apply_progress(self, message: str, percentage: int):
        """Handle progress update from ApplyWorker."""
        self.status_label.setText(message)
        self.progress_bar.setValue(percentage)

    def _on_apply_finished(self):
        """Handle successful completion of ApplyWorker."""
        self.status_label.setText("Apply completed!")
        self.progress_bar.setVisible(False)
        self.apply_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        # Emit registered signal with output paths for Deconvolution
        output_folder = self._apply_output_folder if hasattr(self, '_apply_output_folder') else None
        if output_folder:
            time_min = self.time_min_spin.value()
            output_data = {
                "a_path": os.path.join(output_folder, f"a_t{time_min}.zarr"),
                "b_path": os.path.join(output_folder, f"b_t{time_min}.zarr"),
                "output_folder": output_folder,
            }
            self.registered.emit(output_data)

    def _on_apply_error(self, error_msg: str):
        """Handle error from ApplyWorker."""
        QMessageBox.critical(self, "Error", f"Apply failed: {error_msg}")
        self.status_label.setText("Error during apply")
        self.progress_bar.setVisible(False)
        self.apply_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Batch apply methods
    # ------------------------------------------------------------------

    def _apply_registration_batch(
        self,
        data_path: str,
        output_folder: str,
        time_range: tuple,
        channel_range: tuple,
        multi_pos: bool,
        position: int,
        ignore_bbox: bool,
        save_tiffs: bool,
    ):
        """Submit apply registration as a SLURM batch job."""
        if not self.remote_client:
            self._on_apply_error("Remote client not available")
            return

        params = {
            "data_path": data_path,
            "params_path": os.path.join(data_path, "deskew_registration_params.json"),
            "time_range": list(time_range),
            "channel_range": list(channel_range),
            "multi_pos": multi_pos,
            "position": position,
            "output_folder": output_folder,
            "ignore_bbox": ignore_bbox,
            "save_tiffs": save_tiffs,
        }

        # Disable UI during batch processing
        self.apply_button.setEnabled(False)
        self.load_deskew_button.setEnabled(False)
        self.batch_job_checkbox.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Submitting batch job...")

        try:
            handle = self.remote_client.submit_batch_job(
                command="submit_batch_apply",
                params=params,
                command_type="apply",
            )
            self.status_label.setText(
                f"Batch job #{handle.job_id} submitted — polling..."
            )

            # Connect poller signals
            handle.poller.status_changed.connect(self._on_batch_status_changed)
            handle.poller.finished.connect(self._on_batch_finished)
            handle.poller.error_occurred.connect(self._on_batch_error)
            handle.poller.progress_updated.connect(self._on_batch_progress)

        except Exception as e:
            self.apply_button.setEnabled(True)
            self.load_deskew_button.setEnabled(True)
            self.batch_job_checkbox.setEnabled(True)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(
                self, "Error", f"Failed to submit batch job: {e}"
            )

    def _on_batch_status_changed(self, job_id: str, status: str):
        """Handle status change for batch job."""
        self.status_label.setText(f"Job #{job_id} — {status}")

    def _on_batch_finished(self, result: dict):
        """Handle successful completion of batch apply job."""
        self.apply_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        self.batch_job_checkbox.setEnabled(True)
        self.progress_bar.setVisible(False)

        if result.get("success"):
            output_folder = result.get("result", {}).get(
                "output_folder", "unknown"
            )
            self.status_label.setText("Batch apply complete")
            # Emit registered signal with output paths for Deconvolution
            time_min = self.time_min_spin.value()
            output_data = {
                "a_path": os.path.join(output_folder, f"a_t{time_min}.zarr"),
                "b_path": os.path.join(output_folder, f"b_t{time_min}.zarr"),
                "output_folder": output_folder,
            }
            self.registered.emit(output_data)
        else:
            error_msg = result.get("error", "Unknown error")
            self.status_label.setText("Batch apply failed")
            QMessageBox.critical(self, "Error", f"Apply failed: {error_msg}")

    def _on_batch_error(self, message: str):
        """Handle error from batch job."""
        self.apply_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
        self.batch_job_checkbox.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Batch job error")
        QMessageBox.critical(self, "Error", f"Batch job error: {message}")

    def _on_batch_progress(self, message: str, percentage: int):
        """Handle progress update from batch job poller."""
        self.status_label.setText(message)
        if percentage >= 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(percentage)
