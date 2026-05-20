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
        multi_pos: bool = False,
        time: int = 0,
        position: int = 0,
        auto_crop: bool = True,
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
        self.multi_pos = multi_pos
        self.time = time
        self.position = position
        self.auto_crop = auto_crop

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
            with data.uManagerAcquisition(self.data_path, self.multi_pos, np) as acq:
                if self.multi_pos:
                    volume_a = acq.get(self.position, "a", self.channel, self.time, window=window)
                    volume_b = acq.get(self.position, "b", self.channel, self.time, window=window)
                else:
                    volume_a = acq.get("a", self.channel, self.time, window=window)
                    volume_b = acq.get("b", self.channel, self.time, window=window)

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
                    "auto_crop": self.auto_crop,
                    "preserve_dtype": True,
                    "block_size": (8,8,8),
                }
            elif self.method == "ortho":
                kwargs = {
                    "preserve_dtype": True,
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
            if self.method == "shear":
                kwargs["rotation_thetas"] = (0, math.pi/2, 0)
            b_dsk = dsk.deskew_stage_scan(
                volume_b, self.pixel_size, step_size_lat, -1, 
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
                    "linear",
                    True,
                    (b_dsk.shape[0], b_dsk.shape[1], b_dsk.shape[2]),
                    8, 8, 8
                )
            try:
                b_dsk = b_dsk.get()
            except:
                pass
            # a_dsk = a_dsk.astype(np.float32)
            # b_dsk = b_dsk.astype(np.float32)

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
        metric="cr",
        interp_method="cubspl",
        use_piecewise=True,
        bound_translation=20.0,
        bound_rot_shear=5.0,
        bound_scale=0.05,
    ):
        super().__init__()
        self.a_deskewed = a_deskewed
        self.b_deskewed = b_deskewed
        self.transform_type = transform_type
        self.initial_translation = initial_translation if initial_translation is not None else [0, 0, 0]
        self.metric = metric
        self.interp_method = interp_method
        self.use_piecewise = use_piecewise
        self.bound_translation = bound_translation
        self.bound_rot_shear = bound_rot_shear
        self.bound_scale = bound_scale

    def run(self):
        """Perform registration in background thread."""
        try:
            # Lazy imports to avoid CUDA compilation at module level
            from pyspim.interp import affine
            from pyspim.reg import powell
            from pyspim.util import launch_params_for_volume, pad_to_same_size

            shp_a = self.a_deskewed.shape
            shp_b = self.b_deskewed.shape
            # check if volumes are different size, and pad accordingly
            if shp_a[0] != shp_b[0] or shp_a[1] != shp_b[1] or shp_a[2] != shp_b[2]:
                self.progress_updated.emit("Padding volumes to same size...")
                # Pad volumes to same size
                a_dsk, b_dsk = pad_to_same_size(self.a_deskewed, self.b_deskewed)

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
                T, res = powell.optimize_affine_piecewise(
                    cp.asarray(a_dsk),
                    cp.asarray(b_dsk),
                    metric=self.metric,
                    transform=self.transform_type,
                    interp_method=self.interp_method,
                    par0=par0,
                    bounds=bounds,
                    kernel_launch_params=launch_par,
                    verbose=False,
                )
            else:
                T, res = powell.optimize_affine(
                    cp.asarray(a_dsk),
                    cp.asarray(b_dsk),
                    metric=self.metric,
                    transform=self.transform_type,
                    interp_method=self.interp_method,
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


class ApplyWorker(QThread):
    """Worker thread for applying registration transformations to specified ranges."""

    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(str, int)  # message, percentage (0-100)

    def __init__(
        self,
        data_path: str,
        params_path: str,
        time_range: tuple,
        channel_range: tuple,  # 1-indexed
        multi_pos: bool,
        position: int,
        output_folder: str,
        ignore_bbox: bool,
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

    def _get_deskew_kwargs(self, method: str) -> dict:
        """Get kwargs for deskew based on method, matching LoadDeskewWorker logic."""
        if method == "shear":
            return {
                "rotation_thetas": (0, 0, 0),
                "interp_method": "linear",
                "auto_crop": False,
                "preserve_dtype": True,
                "block_size": (8, 8, 8),
            }
        elif method == "ortho":
            return {
                "preserve_dtype": True,
                "stream": None,
            }
        else:  # dispim
            return {"preserve_dtype": True}

    def _compute_common_crops(self, a_shape, b_shape, t0):
        """Compute crop slices for both volumes to their common overlapping region.

        Replicates the logic from RegistrationWidget._compute_common_crops().

        Args:
            a_shape: Shape of View A deskewed volume (Z, Y, X).
            b_shape: Shape of View B deskewed volume (Z, Y, X).
            t0: Pre-reg translation in pixel units [tz, ty, tx].

        Returns:
            Tuple of (a_slices, b_slices, cropped_shape) or (None, None, None) if no overlap.
        """
        Za, Ya, Xa = a_shape
        Zb, Yb, Xb = b_shape
        tz, ty, tx = t0

        z_start_a = max(0, tz)
        z_end_a = min(Za, tz + Zb)
        y_start_a = max(0, ty)
        y_end_a = min(Ya, ty + Yb)
        x_start_a = max(0, tx)
        x_end_a = min(Xa, tx + Xb)

        if z_start_a >= z_end_a or y_start_a >= y_end_a or x_start_a >= x_end_a:
            return None, None, None

        a_slices = (
            slice(int(z_start_a), int(z_end_a)),
            slice(int(y_start_a), int(y_end_a)),
            slice(int(x_start_a), int(x_end_a)),
        )

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

    def run(self):
        """Apply deskewing and registration to specified time/channel ranges."""
        try:
            import math
            import shutil
            import zarr
            from pyspim.data import dispim as data
            from pyspim import deskew as dsk
            from pyspim.interp import affine

            # Load parameters
            with open(self.params_path, "r") as f:
                params = json.load(f)

            dp = params["deskewing_parameters"]
            method = dp["method"]
            step_size = dp["step_size_um"]
            pixel_size = dp["pixel_size_um"]
            theta_deg = dp["theta_degrees"]
            theta_rad = theta_deg * math.pi / 180
            affine_matrix = np.array(params["affine_registration_matrix"])

            # Load registration parameters for crop_to_common logic
            rp = params["registration_parameters"]
            crop_to_common = rp.get("crop_to_common", False)
            interp_method = rp.get("interp_method", "cubspl")
            pre_reg = params["pre_reg_transform"]
            # Convert pre-reg transform from micrometers to pixel units
            pre_reg_t = [
                pre_reg["tz_um"] / pixel_size,
                pre_reg["ty_um"] / pixel_size,
                pre_reg["tx_um"] / pixel_size,
            ]

            # Calculate lateral step size
            step_size_lat = step_size / math.cos(theta_rad)

            # Get deskew kwargs
            deskew_kwargs = self._get_deskew_kwargs(method)

            # Load bbox
            window = self._load_bbox()

            # Prepare output folder
            if os.path.exists(self.output_folder):
                shutil.rmtree(self.output_folder)
            os.makedirs(self.output_folder, exist_ok=True)

            # Determine channel indices (0-indexed)
            chan_start = self.channel_range[0] - 1
            chan_end = self.channel_range[1] - 1
            channels = list(range(chan_start, chan_end + 1))
            n_channels = len(channels)

            time_start, time_end = self.time_range
            total_items = (time_end - time_start + 1) * len(channels)
            current_item = 0

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

                a_dsk = dsk.deskew_stage_scan(
                    vol_a, pixel_size, step_size_lat, 1,
                    theta=theta_rad, method=method, **deskew_kwargs
                )
                try:
                    a_dsk = a_dsk.get()
                except:
                    pass

                b_dsk = dsk.deskew_stage_scan(
                    vol_b, pixel_size, step_size_lat, -1,
                    theta=theta_rad, method=method, **deskew_kwargs
                )
                try:
                    b_dsk = b_dsk.get()
                except:
                    pass

                # Apply crop_to_common if enabled in saved params
                if crop_to_common:
                    a_slices, b_slices, cropped_shape = self._compute_common_crops(
                        a_dsk.shape, b_dsk.shape, pre_reg_t
                    )
                    if a_slices is not None:
                        a_dsk = a_dsk[a_slices]
                        b_dsk = b_dsk[b_slices]

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
                    f"Time {t}, Channel {first_chan + 1} done ({percentage}%)", percentage
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

                    # Deskew View A
                    a_dsk = dsk.deskew_stage_scan(
                        vol_a, pixel_size, step_size_lat, 1,
                        theta=theta_rad, method=method, **deskew_kwargs
                    )
                    try:
                        a_dsk = a_dsk.get()
                    except:
                        pass

                    # Deskew View B
                    b_dsk = dsk.deskew_stage_scan(
                        vol_b, pixel_size, step_size_lat, -1,
                        theta=theta_rad, method=method, **deskew_kwargs
                    )
                    try:
                        b_dsk = b_dsk.get()
                    except:
                        pass

                    # Apply crop_to_common if enabled in saved params
                    if crop_to_common:
                        a_slices, b_slices, cropped_shape = self._compute_common_crops(
                            a_dsk.shape, b_dsk.shape, pre_reg_t
                        )
                        if a_slices is not None:
                            a_dsk = a_dsk[a_slices]
                            b_dsk = b_dsk[b_slices]

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
                        f"Time {t}, Channel {chan_idx + 1} done ({percentage}%)", percentage
                    )

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

    registered = pyqtSignal(dict)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
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
        # Pre-registration transform state (stored in micrometers: [tz, ty, tx])
        self._pre_reg_transform = [0.0, 0.0, 0.0]
        # Flag to prevent recursive sync when programmatically updating layers
        self._syncing_transforms = False
        # Store pixel size for layer scale
        self._pixel_size = None
        # Store registration results for saving
        self.registration_matrix = None
        self.correlation_ratio = None
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

        # Auto-Crop checkbox (only visible for shear method)
        self.auto_crop_checkbox = QCheckBox("Auto-Crop")
        self.auto_crop_checkbox.setToolTip(
            "When using shear deskewing, automatically crop the output "
            "to remove empty borders introduced by the shearing process."
        )
        self.auto_crop_checkbox.setChecked(True)
        deskew_layout.addRow(self.auto_crop_checkbox)

        # Connect method combo to show/hide auto-crop checkbox
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
        self.channel_min_spin.setRange(1, 20)
        self.channel_min_spin.setValue(1)
        self.channel_max_spin = QSpinBox()
        self.channel_max_spin.setRange(1, 20)
        self.channel_max_spin.setValue(1)
        chan_row.addWidget(self.channel_min_spin)
        chan_row.addWidget(QLabel("-"))
        chan_row.addWidget(self.channel_max_spin)
        apply_layout.addLayout(chan_row)

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

    def _on_method_changed(self, method: str = ""):
        """Handle Method combo box changed - show/hide Auto-Crop checkbox."""
        current_method = method if method else self.method_combo.currentText()
        self.auto_crop_checkbox.setVisible(current_method == "shear")

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
        # Disable apply section when loading new data
        self._set_apply_enabled(False)

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
        theta_deg = self.theta_spin.value()
        theta_rad = theta_deg * math.pi / 180
        method = self.method_combo.currentText()
        auto_crop = self.auto_crop_checkbox.isChecked()

        # Create and start worker
        ignore_bbox = self.ignore_bbox_checkbox.isChecked()
        multi_pos = self.multi_pos_checkbox.isChecked()
        time = self.time_spin.value()
        position = self.position_spin.value()
        self.load_worker = LoadDeskewWorker(
            data_path, channel, projection_type,
            pixel_size, step_size, theta_rad,
            method, ignore_bbox,
            multi_pos, time, position,
            auto_crop,
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
        self.reg_worker = RegistrationWorker(
            a_vol, b_vol, transform_type,
            initial_translation=initial_t,
            metric=metric,
            interp_method=interp_method,
            use_piecewise=use_piecewise,
            bound_translation=bound_translation,
            bound_rot_shear=bound_rot_shear,
            bound_scale=bound_scale,
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
        # Store for saving
        self.registration_matrix = transform_matrix
        self.correlation_ratio = cr

        # Add registered data as new layers
        output_a_name = "View A: Registered"
        output_b_name = "View B: Registered"

        # Remove old layers if they exist
        for layer_name in [output_a_name, output_b_name]:
            try:
                layer = self.viewer.layers[layer_name]
                self.viewer.layers.remove(layer)
            except (KeyError, ValueError):
                pass

        _pixel_size = self.pixel_size_spin.value()
        self.viewer.add_image(
            a_registered,
            name=output_a_name,
            metadata={
                "transform_matrix": transform_matrix,
                "correlation_ratio": cr,
                "transform_type": result["transform_type"],
                "step_size": self.step_size_spin.value(),
                "pixel_size": _pixel_size,
                "theta": self.theta_spin.value() * math.pi / 180,
            },
            scale=(_pixel_size, _pixel_size, _pixel_size),
            opacity=0.5,
            blending="additive",
            colormap="red",
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
            scale=(_pixel_size, _pixel_size, _pixel_size),
            opacity=0.5,
            blending="additive",
            colormap="cyan",
        )

        # Update status and results
        self.status_label.setText(
            f"Registration completed! A: {a_registered.shape}, B: {b_registered.shape}"
        )

        results_text = f"""
        <b>Registration Results:</b><br>
        Metric: {cr:.3f}<br>
        Shape: {a_registered.shape}<br>
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
            "theta": self.theta_spin.value() * math.pi / 180,
        }
        self.registered.emit(output_data)

    def on_error(self, error_msg):
        """Handle error."""
        QMessageBox.critical(self, "Error", f"Operation failed: {error_msg}")
        self.status_label.setText("Error during operation")
        self.load_deskew_button.setEnabled(True)
        self.register_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setVisible(False)

    def save_registration_params(self):
        """Save registration parameters to a JSON file in the Data Path folder."""
        data_path = self.path_edit.text()
        if not data_path or not os.path.exists(data_path):
            QMessageBox.warning(self, "Error", "Please select a valid data path")
            return

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

        # Build parameters dictionary
        params = {
            "deskewing_parameters": {
                "method": self.method_combo.currentText(),
                "step_size_um": self.step_size_spin.value(),
                "pixel_size_um": self.pixel_size_spin.value(),
                "theta_degrees": self.theta_spin.value(),
                "auto_crop": self.auto_crop_checkbox.isChecked(),
            },
            "pre_reg_transform": {
                "tz_um": self._pre_reg_transform[0],
                "ty_um": self._pre_reg_transform[1],
                "tx_um": self._pre_reg_transform[2],
            },
            "registration_parameters": {
                "transform_type": self.transform_combo.currentText(),
                "crop_to_common": self.crop_to_common_checkbox.isChecked(),
                "metric": metric_map.get(self.metric_combo.currentText(), "cr"),
                "interp_method": interp_method_map.get(
                    self.interp_method_combo.currentText(), "cubspl"
                ),
                "use_piecewise": self.piecewise_checkbox.isChecked(),
                "bound_translation": self.bound_translation_spin.value(),
                "bound_rot_shear": self.bound_rot_shear_spin.value(),
                "bound_scale": self.bound_scale_spin.value(),
            },
            "affine_registration_matrix": self.registration_matrix.tolist(),
            "correlation_ratio": self.correlation_ratio,
        }

        # Write JSON file
        output_path = os.path.join(data_path, "deskew_registration_params.json")
        try:
            with open(output_path, "w") as f:
                json.dump(params, f, indent=2)
            self.status_label.setText(f"Registration parameters saved!")
            # Enable Apply section after successful save
            self._update_apply_section()
        except (IOError, OSError) as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters: {e}")

    def _detect_dataset_dimensions(self):
        """Detect the number of timepoints and channels from the dataset.

        Returns:
            Tuple of (n_time, n_channel) where n_channel is the number of
            logical channels (A/B pairs), not raw TIFF channel count.
            Returns (1, 1) if detection fails.
        """
        data_path = self.path_edit.text()
        if not data_path or not os.path.exists(data_path):
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
        """Update Apply section based on existence of deskew_registration_params.json."""
        data_path = self.path_edit.text()
        if not data_path or not os.path.exists(data_path):
            self._set_apply_enabled(False)
            return

        params_path = os.path.join(data_path, "deskew_registration_params.json")
        if os.path.exists(params_path):
            n_time, n_chan = self._detect_dataset_dimensions()
            self.time_min_spin.setRange(0, max(0, n_time - 1))
            self.time_max_spin.setRange(0, max(0, n_time - 1))
            self.time_min_spin.setValue(0)
            self.time_max_spin.setValue(n_time - 1)
            self.channel_min_spin.setRange(1, max(1, n_chan))
            self.channel_max_spin.setRange(1, max(1, n_chan))
            self.channel_min_spin.setValue(1)
            self.channel_max_spin.setValue(n_chan)
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

    def apply_registration(self):
        """Apply saved registration transformations to specified ranges."""
        data_path = self.path_edit.text()
        if not data_path or not os.path.exists(data_path):
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

        # Determine output folder
        if multi_pos:
            output_folder = os.path.join(data_path, f"pos{position}_dskreg")
        else:
            output_folder = os.path.join(data_path, "dskreg")

        # Remove existing output folder
        if os.path.exists(output_folder):
            import shutil
            shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        # Disable UI during processing
        self.apply_button.setEnabled(False)
        self.load_deskew_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting apply...")

        # Create and start worker
        self.apply_worker = ApplyWorker(
            data_path, params_path,
            (time_min, time_max), (chan_min, chan_max),
            multi_pos, position, output_folder, ignore_bbox,
        )
        self.apply_worker.finished.connect(self._on_apply_finished)
        self.apply_worker.error_occurred.connect(self._on_apply_error)
        self.apply_worker.progress_updated.connect(self._on_apply_progress)
        self.apply_worker.start()

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

    def _on_apply_error(self, error_msg: str):
        """Handle error from ApplyWorker."""
        QMessageBox.critical(self, "Error", f"Apply failed: {error_msg}")
        self.status_label.setText("Error during apply")
        self.progress_bar.setVisible(False)
        self.apply_button.setEnabled(True)
        self.load_deskew_button.setEnabled(True)
