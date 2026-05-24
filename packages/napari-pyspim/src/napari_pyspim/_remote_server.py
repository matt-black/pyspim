"""Remote computation server for napari-pyspim.

Reads MessagePack commands from stdin, executes pyspim operations,
and writes MessagePack responses to stdout.

Can be run standalone via: pyspim-remote-server
Or executed by the client after SFTP upload.
"""

import json
import logging
import struct
import sys
import time
import msgpack

# Configure logging to stderr so it doesn't interfere with stdout protocol
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format="%(asctime)s [SERVER] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

START_TIME: float = time.time()

# ---------------------------------------------------------------------------
# Session storage for deskewed volumes
# ---------------------------------------------------------------------------

import uuid

SESSIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def encode_array(obj) -> dict:
    """Encode a numpy array as a MessagePack-serializable dict.

    Uses a ``__numpy__`` marker so the decoder can reconstruct the array.
    """
    import numpy as np  # type: ignore

    if isinstance(obj, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    raise TypeError(f"Cannot encode object of type {type(obj)}")


def decode_array(obj: dict) -> "np.ndarray | dict":  # type: ignore[name-defined]
    """Decode a dict produced by ``encode_array`` back to a numpy array."""
    if isinstance(obj, dict) and obj.get("__numpy__"):
        import numpy as np  # type: ignore

        return np.frombuffer(
            obj["data"], dtype=np.dtype(obj["dtype"])
        ).reshape(obj["shape"])
    return obj


# ---------------------------------------------------------------------------
# Message framing  (length-prefixed MessagePack)
# ---------------------------------------------------------------------------

def send_response(stdout, request_id, result, error=None):
    """Serialize and write a response message to *stdout*."""
    msg = {
        "request_id": request_id,
        "success": error is None,
        "result": result,
        "error": error,
    }
    payload = msgpack.packb(msg, default=encode_array)
    header = struct.pack(">Q", len(payload))  # 8-byte big-endian uint64
    stdout.buffer.write(header + payload)
    stdout.buffer.flush()


def send_progress(stdout, request_id, message, percentage=None):
    """Serialize and write a progress update to *stdout*."""
    msg = {
        "request_id": request_id,
        "progress": message,
        "percentage": percentage,
    }
    payload = msgpack.packb(msg)
    header = struct.pack(">Q", len(payload))
    stdout.buffer.write(header + payload)
    stdout.buffer.flush()


def receive_message(stdin_buffer):
    """Read a length-prefixed MessagePack message from *stdin_buffer*."""
    header = _read_exact(stdin_buffer, 8)
    if len(header) < 8:
        raise EOFError("Channel closed by client")
    length = struct.unpack(">Q", header)[0]
    data = _read_exact(stdin_buffer, length)
    return msgpack.unpackb(data, strict_map_key=False, object_hook=decode_array)


def _read_exact(buffer, n: int) -> bytes:
    """Read exactly *n* bytes from *buffer*, blocking if necessary."""
    data = b""
    while len(data) < n:
        chunk = buffer.read(n - len(data))
        if not chunk:
            break
        data += chunk
    return data


# ---------------------------------------------------------------------------
# Command handlers  (stubs -- implemented in later phases)
# ---------------------------------------------------------------------------

def handle_ping(params: dict) -> dict:
    """Health-check handler."""
    return {"status": "ok", "uptime": time.time() - START_TIME}


def handle_compute(params: dict) -> dict:
    """Simple calculation test to verify connection health.

    Evaluates the expression provided in params (defaulting to "2+2")
    and returns the result.
    """
    expression = params.get("expression", "2+2")
    result = eval(expression)  # noqa: S307  # Controlled test expression
    return {"expression": expression, "result": result}


def handle_compute_projections(params: dict) -> dict:
    """Load data and compute 2D projections for ROI detection.

    Parameters
    ----------
    params : dict
        data_path : str
        channel : int  (0-indexed)
        projection_type : str  ('max', 'sum', 'mean')
        multi_pos : bool
        time : int
        position : int

    Returns
    -------
    dict with keys: yx_proj_a, zy_proj_a, yx_proj_b, zy_proj_b,
                    volume_shape_a, volume_shape_b
    """
    import numpy as np

    data_path = params["data_path"]
    channel = params["channel"]
    projection_type = params["projection_type"]
    multi_pos = params.get("multi_pos", False)
    time = params.get("time", 0)
    position = params.get("position", 0)

    from pyspim.data import dispim as data

    with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
        if multi_pos:
            volume_a = acq.get(position, "a", channel, time)
            volume_b = acq.get(position, "b", channel, time)
        else:
            volume_a = acq.get("a", channel, time)
            volume_b = acq.get("b", channel, time)

    # Compute projections (only YX and ZY needed for ROI detection)
    yx_proj_a, zy_proj_a, _ = _compute_projections(volume_a, projection_type)
    yx_proj_b, zy_proj_b, _ = _compute_projections(volume_b, projection_type)

    return {
        "yx_proj_a": yx_proj_a,
        "zy_proj_a": zy_proj_a,
        "yx_proj_b": yx_proj_b,
        "zy_proj_b": zy_proj_b,
        "volume_shape_a": list(volume_a.shape),
        "volume_shape_b": list(volume_b.shape),
    }


def _compute_projections(volume, proj_type):
    """Compute YX, ZY, and XZ projections from a volume."""
    try:
        import cupy as np
        _HAS_CUPY = True
    except:
        import numpy as np
        _HAS_CUPY = False

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
    if _HAS_CUPY:
        return yx_proj.get(), zy_proj.get(), xz_proj.get()
    else:
        return yx_proj, zy_proj, xz_proj


def handle_query_positions(params: dict) -> dict:
    """Query the number of positions in a multi-position acquisition.

    Parameters
    ----------
    params : dict
        data_path : str

    Returns
    -------
    dict with key: num_positions
    """
    import numpy as np

    data_path = params["data_path"]

    from pyspim.data import dispim as data

    with data.uManagerAcquisition(data_path, True, np) as acq:
        num_positions = acq.num_positions

    return {"num_positions": num_positions}


def handle_load_deskew(params: dict) -> dict:
    """Load data, deskew, and compute projections.

    Mirrors ``LoadDeskewWorker.run()`` from ``_registration.py``.

    Parameters
    ----------
    params : dict
        data_path : str
        channel : int  (0-indexed)
        projection_type : str  ('max', 'sum', 'mean')
        pixel_size : float
        step_size : float
        theta : float  (in radians)
        method : str  ('orthogonal', 'dispim', 'shear')
        ignore_bbox : bool
        multi_pos : bool
        time : int
        position : int
        auto_crop : bool

    Returns
    -------
    dict with keys: session_id, yx_proj_a, zy_proj_a, xz_proj_a,
                    yx_proj_b, zy_proj_b, xz_proj_b,
                    volume_shape_a, volume_shape_b, method, step_size_lat
    """
    import math
    try:
        import cupy as np
    except:
        import numpy as np
    import os

    data_path = params["data_path"]
    channel = params["channel"]
    projection_type = params["projection_type"]
    pixel_size = params["pixel_size"]
    step_size = params["step_size"]
    theta = params["theta"]
    method = params["method"]
    ignore_bbox = params.get("ignore_bbox", False)
    multi_pos = params.get("multi_pos", False)
    time = params.get("time", 0)
    position = params.get("position", 0)
    auto_crop = params.get("auto_crop", True)

    from pyspim.data import dispim as data
    from pyspim import deskew as dsk

    # --- Load bounding box if present ---
    window = None
    if not ignore_bbox:
        bbox_path = os.path.join(data_path, "bbox_raw.json")
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, "r") as f:
                    bbox = json.load(f)
                window = (
                    slice(bbox[0][0], bbox[0][1]),
                    slice(bbox[1][0], bbox[1][1]),
                    slice(bbox[2][0], bbox[2][1]),
                )
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

    # --- Load data ---
    with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
        if multi_pos:
            volume_a = acq.get(position, "a", channel, time, window=window)
            volume_b = acq.get(position, "b", channel, time, window=window)
        else:
            volume_a = acq.get("a", channel, time, window=window)
            volume_b = acq.get("b", channel, time, window=window)

    # --- Calculate lateral step size ---
    step_size_lat = step_size / math.cos(theta)

    # --- Deskew View A (direction = 1) ---
    if method == "shear":
        deskew_kwargs_a = {
            "rotation_thetas": (0, 0, 0),
            "interp_method": "linear",
            "auto_crop": auto_crop,
            "preserve_dtype": True,
            "block_size": (8, 8, 8),
        }
    elif method == "ortho":
        deskew_kwargs_a = {
            "preserve_dtype": True,
            "stream": None,
        }
    else:
        deskew_kwargs_a = {}

    a_dsk = dsk.deskew_stage_scan(
        volume_a, pixel_size, step_size_lat, 1,
        theta=theta, method=method, **deskew_kwargs_a,
    )
    yx_proj_a, zy_proj_a, xz_proj_a = _compute_projections(a_dsk, projection_type)
    try:
        a_dsk = a_dsk.get()
    except Exception:
        pass
    
    # --- Deskew View B (direction = -1) ---
    if method == "shear":
        deskew_kwargs_b = {
            "rotation_thetas": (0, math.pi / 2, 0),
            "interp_method": "linear",
            "auto_crop": auto_crop,
            "preserve_dtype": True,
            "block_size": (8, 8, 8),
        }
    elif method == "ortho":
        deskew_kwargs_b = {
            "preserve_dtype": True,
            "stream": None,
        }
    else:
        deskew_kwargs_b = {}

    b_dsk = dsk.deskew_stage_scan(
        volume_b, pixel_size, step_size_lat, -1,
        theta=theta, method=method, **deskew_kwargs_b,
    )

    if method == "dispim":
        # Rotate the volume in the XZ plane so that it is in the same
        # coordinate system as that of view A
        from pyspim.interp.affine import transform as affine_transform
        from pyspim._matrix import rotation_about_point_matrix

        R = rotation_about_point_matrix(
            0, math.pi / 2, 0,
            *[s / 2 for s in b_dsk.shape[::-1]],
        )
        b_dsk = affine_transform(
            b_dsk, R,
            "linear", True,
            (b_dsk.shape[0], b_dsk.shape[1], b_dsk.shape[2]),
            8, 8, 8,
        )

    # --- Compute projections ---
    yx_proj_b, zy_proj_b, xz_proj_b = _compute_projections(b_dsk, projection_type)

    # --- Store deskewed volumes on server ---
    session_id = str(uuid.uuid4())

    try:
        b_dsk = b_dsk.get()
    except Exception:
        pass

    SESSIONS[session_id] = {
        "a_deskewed": a_dsk,
        "b_deskewed": b_dsk,
        "volume_shape_a": a_dsk.shape,
        "volume_shape_b": b_dsk.shape,
        "method": method,
        "step_size_lat": step_size_lat,
        "pixel_size": pixel_size,
        "theta": theta,
    }

    return {
        "session_id": session_id,
        "yx_proj_a": yx_proj_a,
        "zy_proj_a": zy_proj_a,
        "xz_proj_a": xz_proj_a,
        "yx_proj_b": yx_proj_b,
        "zy_proj_b": zy_proj_b,
        "xz_proj_b": xz_proj_b,
        "volume_shape_a": list(a_dsk.shape),
        "volume_shape_b": list(b_dsk.shape),
        "method": method,
        "step_size_lat": step_size_lat,
    }


def _normalize_deskew_method(method: str) -> str:
    """Normalize deskew method name from UI display name to pyspim internal name."""
    method_map = {
        "orthogonal": "ortho",
        "ortho": "ortho",
        "dispim": "dispim",
        "shear": "shear",
    }
    return method_map.get(method.lower(), method)


def _get_deskew_kwargs(method: str) -> dict:
    """Get kwargs for deskew based on method, matching ApplyWorker logic."""
    import math

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


def _compute_common_crops_server(a_shape, b_shape, t0):
    """Compute crop slices for both volumes to their common overlapping region.

    Server-side mirror of ``RegistrationWidget._compute_common_crops()``.

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

    z_start_b = max(0, -tz)
    z_end_b = min(Zb, Za - tz)
    y_start_b = max(0, -ty)
    y_end_b = min(Yb, Ya - ty)
    x_start_b = max(0, -tx)
    x_end_b = min(Xb, Xa - tx)

    shape_a = (
        int(z_end_a - z_start_a),
        int(y_end_a - y_start_a),
        int(x_end_a - x_start_a),
    )
    shape_b = (
        int(z_end_b - z_start_b),
        int(y_end_b - y_start_b),
        int(x_end_b - x_start_b),
    )

    cropped_shape = tuple(min(a, b) for a, b in zip(shape_a, shape_b))

    a_slices = (
        slice(int(z_start_a), int(z_start_a) + cropped_shape[0]),
        slice(int(y_start_a), int(y_start_a) + cropped_shape[1]),
        slice(int(x_start_a), int(x_start_a) + cropped_shape[2]),
    )

    b_slices = (
        slice(int(z_start_b), int(z_start_b) + cropped_shape[0]),
        slice(int(y_start_b), int(y_start_b) + cropped_shape[1]),
        slice(int(x_start_b), int(x_start_b) + cropped_shape[2]),
    )

    return a_slices, b_slices, cropped_shape


def handle_register(params: dict) -> dict:
    """Register two deskewed volumes stored in a session.

    Mirrors ``RegistrationWorker.run()`` from ``_registration.py`` but
    operates on volumes already stored in ``SESSIONS`` (from a prior
    ``load_deskew`` call).  Returns only the transform matrix and
    correlation ratio, not the registered volumes themselves.

    Parameters
    ----------
    params : dict
        session_id : str
        transform_type : str  ('t', 't+r', 't+r+s', 't+sh', 't+ssh', 't+sh+s')
        pre_reg_translation : list  [tz, ty, tx] in pixel units
        crop_to_common : bool
        metric : str  ('nip', 'cr', 'ncc')
        interp_method : str  ('linear', 'cubspl')
        use_piecewise : bool
        bound_translation : float
        bound_rot_shear : float
        bound_scale : float

    Returns
    -------
    dict with keys: transform_matrix (nested list), correlation_ratio (float)
    """
    import math
    import numpy as np
    try:
        import cupy as cp
    except ImportError:
        cp = np  # fallback to CPU

    session_id = params["session_id"]
    transform_type = params["transform_type"]
    pre_reg_translation = params["pre_reg_translation"]  # [tz, ty, tx] in pixels
    crop_to_common = params.get("crop_to_common", False)
    metric = params.get("metric", "cr")
    interp_method = params.get("interp_method", "cubspl")
    use_piecewise = params.get("use_piecewise", True)
    bound_translation = params.get("bound_translation", 20.0)
    bound_rot_shear = params.get("bound_rot_shear", 5.0)
    bound_scale = params.get("bound_scale", 0.05)

    # --- Retrieve deskewed volumes from session ---
    if session_id not in SESSIONS:
        raise KeyError(f"Session '{session_id}' not found on server")

    session = SESSIONS[session_id]
    a_dsk = session["a_deskewed"]
    b_dsk = session["b_deskewed"]

    # --- Apply crop_to_common if requested ---
    t0 = pre_reg_translation
    if crop_to_common:
        a_slices, b_slices, cropped_shape = _compute_common_crops_server(
            a_dsk.shape, b_dsk.shape, t0
        )
        if a_slices is not None:
            a_dsk = a_dsk[a_slices]
            b_dsk = b_dsk[b_slices]
            t0 = [0, 0, 0]  # Pre-reg translation accounted for by crop

    # --- Pad volumes to same size if needed ---
    from pyspim.util import pad_to_same_size
    shp_a = a_dsk.shape
    shp_b = b_dsk.shape
    if shp_a[0] != shp_b[0] or shp_a[1] != shp_b[1] or shp_a[2] != shp_b[2]:
        a_dsk, b_dsk = pad_to_same_size(a_dsk, b_dsk)

    # --- Set up initial parameters and bounds ---
    bt = bound_translation
    br = bound_rot_shear
    bs = bound_scale

    if transform_type == "t":
        par0 = list(t0)
        bounds = [(t - bt, t + bt) for t in t0]
    elif transform_type == "t+r":
        par0 = list(t0) + [0, 0, 0]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == "t+r+s":
        par0 = list(t0) + [0, 0, 0] + [1, 1, 1]
        bounds = (
            [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3 + [(1 - bs, 1 + bs)] * 3
        )
    elif transform_type == "t+sh":
        par0 = list(t0) + [0, 0, 0, 0, 0, 0]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6
    elif transform_type == "t+ssh":
        par0 = list(t0) + [0, 0, 0]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == "t+sh+s":
        par0 = list(t0) + [0, 0, 0, 0, 0, 0] + [1, 1, 1]
        bounds = (
            [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6 + [(1 - bs, 1 + bs)] * 3
        )
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    par0 = np.array(par0, dtype=np.float64)

    # --- Perform optimization ---
    from pyspim.reg import powell
    from pyspim.util import launch_params_for_volume

    launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)

    a_gpu = cp.asarray(a_dsk)
    b_gpu = cp.asarray(b_dsk)

    if use_piecewise:
        T, res = powell.optimize_affine_piecewise(
            a_gpu,
            b_gpu,
            metric=metric,
            transform=transform_type,
            interp_method=interp_method,
            par0=par0,
            bounds=bounds,
            kernel_launch_params=launch_par,
            verbose=False,
        )
    else:
        T, res = powell.optimize_affine(
            a_gpu,
            b_gpu,
            metric=metric,
            transform=transform_type,
            interp_method=interp_method,
            par0=par0,
            bounds=bounds,
            kernel_launch_params=launch_par,
            verbose=False,
        )

    # --- Extract results ---
    if hasattr(T, "get"):
        T = T.get()
    if hasattr(T, "tolist"):
        T_list = T.tolist()
    else:
        T_list = T

    cr = 1 - res.fun

    return {
        "transform_matrix": T_list,
        "correlation_ratio": cr,
    }


def handle_save_params(params: dict) -> dict:
    """Save registration parameters to a JSON file on the remote server.

    Parameters
    ----------
    params : dict
        data_path : str
            Remote path to the data folder.
        params_dict : dict
            The parameters dictionary to serialize as JSON.

    Returns
    -------
    dict with key: output_path (str)
    """
    import os

    data_path = params["data_path"]
    params_dict = params["params_dict"]

    output_path = os.path.join(data_path, "deskew_registration_params.json")
    with open(output_path, "w") as f:
        json.dump(params_dict, f, indent=2)

    return {"output_path": output_path}


def handle_deconvolve(params: dict) -> dict:
    """Dual-view Richardson-Lucy deconvolution.

    Mirrors ``DeconvolutionWorker.run()`` from ``_deconvolution.py``.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("deconvolve not yet implemented")


def handle_apply_registration(params: dict) -> dict:
    """Batch-apply deskew + registration across time/channel dimensions.

    Mirrors ApplyWorker.run() from _registration.py.

    Parameters
    ----------
    params : dict
        data_path : str
            Path to the μManager acquisition folder on the remote server.
        output_folder : str
            Path where output zarr files will be saved.
        time_range : tuple
            (time_min, time_max) inclusive range of timepoints.
        channel_range : tuple
            (chan_min, chan_max) 1-indexed inclusive range of channels.
        multi_pos : bool
            Whether this is a multi-position acquisition.
        position : int
            Position index for multi-position acquisitions.
        ignore_bbox : bool
            If True, ignore bbox_raw.json and load full dataset.

    Returns
    -------
    dict with keys: output_folder, files_created (list of paths)
    """
    import math
    import os
    import shutil

    import numpy as np
    import zarr

    try:
        import cupy as cp
    except ImportError:
        cp = np

    data_path = params["data_path"]
    output_folder = params["output_folder"]
    time_range = params["time_range"]
    channel_range = params["channel_range"]
    multi_pos = params.get("multi_pos", False)
    position = params.get("position", 0)
    ignore_bbox = params.get("ignore_bbox", False)
    request_id = params.get("_request_id", 0)

    from pyspim.data import dispim as data
    from pyspim import deskew as dsk
    from pyspim.interp import affine

    # Load parameters from saved JSON file
    params_path = os.path.join(data_path, "deskew_registration_params.json")
    with open(params_path, "r") as f:
        saved_params = json.load(f)

    dp = saved_params["deskewing_parameters"]
    method = _normalize_deskew_method(dp["method"])
    step_size = dp["step_size_um"]
    pixel_size = dp["pixel_size_um"]
    theta_deg = dp["theta_degrees"]
    theta_rad = theta_deg * math.pi / 180
    affine_matrix = np.array(saved_params["affine_registration_matrix"])

    rp = saved_params["registration_parameters"]
    crop_to_common = rp.get("crop_to_common", False)
    interp_method = rp.get("interp_method", "cubspl")
    pre_reg = saved_params["pre_reg_transform"]
    pre_reg_t = [
        pre_reg["tz_um"] / pixel_size,
        pre_reg["ty_um"] / pixel_size,
        pre_reg["tx_um"] / pixel_size,
    ]

    step_size_lat = step_size / math.cos(theta_rad)
    deskew_kwargs = _get_deskew_kwargs(method)

    # Load bbox
    window = None
    if not ignore_bbox:
        bbox_path = os.path.join(data_path, "bbox_raw.json")
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, "r") as f:
                    bbox = json.load(f)
                window = (
                    slice(bbox[0][0], bbox[0][1]),
                    slice(bbox[1][0], bbox[1][1]),
                    slice(bbox[2][0], bbox[2][1]),
                )
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

    # Prepare output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Determine channel indices (0-indexed)
    chan_start = channel_range[0] - 1
    chan_end = channel_range[1] - 1
    channels = list(range(chan_start, chan_end + 1))
    n_channels = len(channels)

    time_start, time_end = time_range
    total_items = (time_end - time_start + 1) * len(channels)
    current_item = 0
    files_created = []

    for t in range(time_start, time_end + 1):
        # Process first channel to determine output shape
        first_chan = channels[0]
        send_progress(
            sys.stdout, request_id,
            f"Processing Time {t}, Channel {first_chan + 1} (determining shape)...", 0,
        )

        with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
            if multi_pos:
                vol_a = acq.get(position, "a", first_chan, t, window=window)
                vol_b = acq.get(position, "b", first_chan, t, window=window)
            else:
                vol_a = acq.get("a", first_chan, t, window=window)
                vol_b = acq.get("b", first_chan, t, window=window)

        a_dsk = dsk.deskew_stage_scan(
            vol_a, pixel_size, step_size_lat, 1,
            theta=theta_rad, method=method, **deskew_kwargs,
        )
        try:
            a_dsk = a_dsk.get()
        except Exception:
            pass

        b_dsk = dsk.deskew_stage_scan(
            vol_b, pixel_size, step_size_lat, -1,
            theta=theta_rad, method=method, **deskew_kwargs,
        )
        try:
            b_dsk = b_dsk.get()
        except Exception:
            pass

        if crop_to_common:
            a_slices, b_slices, cropped_shape = _compute_common_crops_server(
                a_dsk.shape, b_dsk.shape, pre_reg_t,
            )
            if a_slices is not None:
                a_dsk = a_dsk[a_slices]
                b_dsk = b_dsk[b_slices]

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

        min_shape = tuple(min(a, b) for a, b in zip(a_dsk.shape, b_reg.shape))
        out_shape = (n_channels, *min_shape)

        a_zarr_path = os.path.join(output_folder, f"a_t{t}.zarr")
        b_zarr_path = os.path.join(output_folder, f"b_t{t}.zarr")

        arr_a = zarr.open(
            a_zarr_path, mode="w", shape=out_shape,
            dtype=np.uint16, chunks=(1, 64, 256, 256),
        )
        arr_b = zarr.open(
            b_zarr_path, mode="w", shape=out_shape,
            dtype=np.uint16, chunks=(1, 64, 256, 256),
        )

        a_final = a_dsk[: min_shape[0], : min_shape[1], : min_shape[2]]
        b_final = b_reg[: min_shape[0], : min_shape[1], : min_shape[2]]
        arr_a[0, ...] = a_final.astype(np.uint16)
        arr_b[0, ...] = b_final.astype(np.uint16)

        files_created.extend([a_zarr_path, b_zarr_path])

        current_item += 1
        percentage = int((current_item / total_items) * 100)
        send_progress(
            sys.stdout, request_id,
            f"Time {t}, Channel {first_chan + 1} done ({percentage}%)", percentage,
        )

        # Process remaining channels
        for chan_idx in channels[1:]:
            c = chan_idx - chan_start

            with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
                if multi_pos:
                    vol_a = acq.get(position, "a", chan_idx, t, window=window)
                    vol_b = acq.get(position, "b", chan_idx, t, window=window)
                else:
                    vol_a = acq.get("a", chan_idx, t, window=window)
                    vol_b = acq.get("b", chan_idx, t, window=window)

            a_dsk = dsk.deskew_stage_scan(
                vol_a, pixel_size, step_size_lat, 1,
                theta=theta_rad, method=method, **deskew_kwargs,
            )
            try:
                a_dsk = a_dsk.get()
            except Exception:
                pass

            b_dsk = dsk.deskew_stage_scan(
                vol_b, pixel_size, step_size_lat, -1,
                theta=theta_rad, method=method, **deskew_kwargs,
            )
            try:
                b_dsk = b_dsk.get()
            except Exception:
                pass

            if crop_to_common:
                a_slices, b_slices, cropped_shape = _compute_common_crops_server(
                    a_dsk.shape, b_dsk.shape, pre_reg_t,
                )
                if a_slices is not None:
                    a_dsk = a_dsk[a_slices]
                    b_dsk = b_dsk[b_slices]

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

            a_final = a_dsk[: min_shape[0], : min_shape[1], : min_shape[2]]
            b_final = b_reg[: min_shape[0], : min_shape[1], : min_shape[2]]

            arr_a[c, ...] = a_final.astype(np.uint16)
            arr_b[c, ...] = b_final.astype(np.uint16)

            current_item += 1
            percentage = int((current_item / total_items) * 100)
            send_progress(
                sys.stdout, request_id,
                f"Time {t}, Channel {chan_idx + 1} done ({percentage}%)", percentage,
            )

    return {"output_folder": output_folder, "files_created": files_created}


def handle_save_zarr(params: dict) -> dict:
    """Move a zarr archive from a temp path to a permanent location.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("save_zarr not yet implemented")


def handle_cleanup_zarr(params: dict) -> dict:
    """Remove a temporary zarr directory.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("cleanup_zarr not yet implemented")


COMMAND_HANDLERS = {
    "ping": handle_ping,
    "compute": handle_compute,
    "compute_projections": handle_compute_projections,
    "query_positions": handle_query_positions,
    "load_deskew": handle_load_deskew,
    "register": handle_register,
    "save_params": handle_save_params,
    "deconvolve": handle_deconvolve,
    "apply_registration": handle_apply_registration,
    "save_zarr": handle_save_zarr,
    "cleanup_zarr": handle_cleanup_zarr,
}


# ---------------------------------------------------------------------------
# Capability detection helpers
# ---------------------------------------------------------------------------

def _check_cuda() -> bool:
    """Return True if a CUDA device is available via cupy."""
    try:
        import cupy  # type: ignore
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _get_pyspim_version() -> str:
    """Return the installed pyspim version string, or 'unknown'."""
    try:
        from importlib.metadata import version
        # import pyspim  # type: ignore
        return version('pyspim')
    except Exception:
        return "not installed"


# ---------------------------------------------------------------------------
# Main server loop
# ---------------------------------------------------------------------------

def main():
    """Entry point for the remote server.

    Sends a ``ready`` message, then enters the command dispatch loop.
    Exits cleanly on ``shutdown`` command or when stdin is closed.
    """
    stdout = sys.stdout
    stdin_buffer = sys.stdin.buffer

    logger.info("Server starting up...")

    # Send ready message
    try:
        logger.info("Checking CUDA availability...")
        has_cuda = _check_cuda()
        logger.info("CUDA available: %s", has_cuda)
    except Exception as e:
        logger.error("CUDA check failed: %s", e, exc_info=True)
        has_cuda = False

    try:
        logger.info("Checking pyspim version...")
        pyspim_version = _get_pyspim_version()
        logger.info("pyspim version: %s", pyspim_version)
    except Exception as e:
        logger.error("pyspim version check failed: %s", e, exc_info=True)
        pyspim_version = "error"

    ready = {
        "type": "ready",
        "version": "0.1.0",
        "capabilities": {
            "has_cuda": has_cuda,
            "pyspim_version": pyspim_version,
        },
    }
    logger.info("Sending ready message: %s", ready)
    payload = msgpack.packb(ready, default=encode_array)
    header = struct.pack(">Q", len(payload))
    stdout.buffer.write(header + payload)
    stdout.buffer.flush()
    logger.info("Ready message sent, entering command loop")

    # Main command loop
    while True:
        try:
            logger.info("Waiting for incoming message...")
            message = receive_message(stdin_buffer)
            logger.info("Received message: request_id=%s, command=%s",
                        message.get("request_id"), message.get("command"))
        except (EOFError, ConnectionError) as e:
            logger.info("Client disconnected (EOF/ConnectionError): %s", e)
            break  # Client disconnected
        except Exception as e:
            logger.error("Unexpected error receiving message: %s", e, exc_info=True)
            break

        request_id = message.get("request_id")
        command = message.get("command")
        params = message.get("params", {})

        if command == "shutdown":
            logger.info("Shutdown command received, responding and exiting")
            send_response(stdout, request_id, {"status": "shutting down"})
            break

        handler = COMMAND_HANDLERS.get(command)
        if handler is None:
            logger.warning("Unknown command: %s", command)
            send_response(stdout, request_id, None,
                          error=f"Unknown command: {command}")
            continue

        try:
            logger.info("Executing handler for command: %s", command)
            # Inject request_id into params so handlers can emit progress updates
            params_with_id = dict(params)
            params_with_id["_request_id"] = request_id
            result = handler(params_with_id)
            logger.info("Handler completed, sending response for request_id=%s", request_id)
            send_response(stdout, request_id, result)
        except Exception as e:
            logger.error("Handler exception for command %s: %s", command, e, exc_info=True)
            import traceback
            traceback.print_exc()
            send_response(stdout, request_id, None,
                          error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
