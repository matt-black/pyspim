"""Remote computation server for napari-pyspim.

Reads MessagePack commands from stdin, executes pyspim operations,
and writes MessagePack responses to stdout.

Can be run standalone via: pyspim-remote-server
Or executed by the client after SFTP upload.
"""
import os
import sys
import time
import json
import shutil
import struct
import logging
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
        import cupy
        if cupy.get_array_module(volume) == cupy:
            np = cupy
            _HAS_CUPY = True
        else:
            import numpy as np
            _HAS_CUPY = False
    except:
        import numpy as np
        _HAS_CUPY = False

    if proj_type == "max":
        yx_proj = np.max(volume, axis=0)  # shape: (Y, X)
        yx_proj = yx_proj.get() if _HAS_CUPY else yx_proj
        zy_proj = np.max(volume, axis=2)  # shape: (Z, Y)
        zy_proj = zy_proj.get() if _HAS_CUPY else zy_proj
        xz_proj = np.max(volume, axis=1)  # shape: (Z, X)
        xz_proj = xz_proj.get() if _HAS_CUPY else xz_proj
    elif proj_type == "sum":
        yx_proj = np.sum(volume, axis=0)
        yx_proj = yx_proj.get() if _HAS_CUPY else yx_proj
        zy_proj = np.sum(volume, axis=2)
        zy_proj = zy_proj.get() if _HAS_CUPY else zy_proj
        xz_proj = np.sum(volume, axis=1)
        xz_proj = xz_proj.get() if _HAS_CUPY else xz_proj
    elif proj_type == "mean":
        yx_proj = np.mean(volume, axis=0)
        yx_proj = yx_proj.get() if _HAS_CUPY else yx_proj
        zy_proj = np.mean(volume, axis=2)
        zy_proj = zy_proj.get() if _HAS_CUPY else zy_proj
        xz_proj = np.mean(volume, axis=1)
        xz_proj = xz_proj.get() if _HAS_CUPY else xz_proj
    else:
        raise ValueError(f"Unknown projection type: {proj_type}")
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
                    volume_shape_a, volume_shape_b, method, step_size
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
    theta = math.pi / 4  # Hardcoded to 45 degrees
    method = params["method"]
    ignore_bbox = params.get("ignore_bbox", False)
    multi_pos = params.get("multi_pos", False)
    time = params.get("time", 0)
    position = params.get("position", 0)
    auto_crop = params.get("auto_crop", True)
    camera_offset = params.get("camera_offset", 0)

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

    # --- Subtract camera offset if non-zero ---
    if camera_offset > 0:
        from pyspim.data.dispim import subtract_constant_uint16arr
        volume_a = subtract_constant_uint16arr(volume_a, camera_offset)
        volume_b = subtract_constant_uint16arr(volume_b, camera_offset)

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
    elif method == "affine":
        deskew_kwargs_a = {
            "preserve_dtype": True,
            "interp_method": "cubspl",
            "block_size": (8,8,8),
        }
    else:
        deskew_kwargs_a = {}

    a_dsk = dsk.deskew_stage_scan(
        volume_a, pixel_size, step_size, 1,
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
    elif method == "affine":
        deskew_kwargs_b = {
            "preserve_dtype": True,
            "interp_method": "cubspl",
            "block_size": (8,8,8),
        }
    else:
        deskew_kwargs_b = {}

    b_dsk = dsk.deskew_stage_scan(
        volume_b, pixel_size, step_size, -1,
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
        "step_size": step_size,
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
        "step_size": step_size,
    }


def _normalize_deskew_method(method: str) -> str:
    """Normalize deskew method name from UI display name to pyspim internal name."""
    method_map = {
        "orthogonal": "ortho",
        "ortho": "ortho",
        "dispim": "dispim",
        "shear": "shear",
        "affine": "affine",
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
    elif method == "affine":
        return {
            "preserve_dtype": True,
            "interp_method": "cubspl",
            "block_size": (8, 8, 8),
        }
    else:  # dispim
        return {"preserve_dtype": True}


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
        crop_bounds : dict with z_start, z_end, y_start, y_end, x_start, x_end
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
    crop_bounds = params.get("crop_bounds")
    metric = params.get("metric", "cr")
    interp_method = params.get("interp_method", "cubspl")
    use_piecewise = params.get("use_piecewise", True)
    bound_translation = params.get("bound_translation", 20.0)
    bound_rot_shear = params.get("bound_rot_shear", 5.0)
    bound_scale = params.get("bound_scale", 0.05)
    opt_method = params.get("opt_method", "powell")

    # --- Retrieve deskewed volumes from session ---
    if session_id not in SESSIONS:
        raise KeyError(f"Session '{session_id}' not found on server")

    session = SESSIONS[session_id]
    a_dsk = session["a_deskewed"]
    b_dsk = session["b_deskewed"]

    # --- Apply crop bounds if present ---
    t0 = [0, 0, 0]  # Pre-reg translation absorbed by crop
    if crop_bounds:
        z_start = crop_bounds["z_start"]
        z_end = crop_bounds["z_end"]
        y_start = crop_bounds["y_start"]
        y_end = crop_bounds["y_end"]
        x_start = crop_bounds["x_start"]
        x_end = crop_bounds["x_end"]
        tz, ty, tx = pre_reg_translation

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
    from pyspim.reg import opt
    from pyspim.util import launch_params_for_volume

    launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)

    a_gpu = cp.asarray(a_dsk)
    b_gpu = cp.asarray(b_dsk)

    if use_piecewise:
        T, res = opt.optimize_affine_piecewise(
            a_gpu,
            b_gpu,
            metric=metric,
            transform=transform_type,
            interp_method=interp_method,
            opt_method=opt_method,
            par0=par0,
            bounds=bounds,
            kernel_launch_params=launch_par,
            verbose=False,
        )
    else:
        T, res = opt.optimize_affine(
            a_gpu,
            b_gpu,
            metric=metric,
            transform=transform_type,
            interp_method=interp_method,
            opt_method=opt_method,
            par0=par0,
            bounds=bounds,
            kernel_launch_params=launch_par,
            verbose=False,
        )

    # --- Apply transformation to B ---
    from pyspim.interp import affine
    b_reg = affine.transform(
        cp.asarray(b_dsk), cp.asarray(T),
        interp_method=interp_method, preserve_dtype=True,
        out_shp=None, block_size_z=8, block_size_y=8, block_size_x=8,
    ).get()

    # Crop to common size (matching local worker logic)
    min_shape = tuple(min(a, b) for a, b in zip(a_dsk.shape, b_reg.shape))
    a_cropped = a_dsk[:min_shape[0], :min_shape[1], :min_shape[2]]
    b_cropped = b_reg[:min_shape[0], :min_shape[1], :min_shape[2]]

    # Compute max projections for registered B
    yx_proj_b = np.max(b_cropped, axis=0)  # (Y, X)
    zy_proj_b = np.max(b_cropped, axis=2)  # (Z, Y)
    xz_proj_b = np.max(b_cropped, axis=1)  # (Z, X)

    # Always compute max projections for cropped A
    yx_proj_a = np.max(a_cropped, axis=0)  # (Y, X)
    zy_proj_a = np.max(a_cropped, axis=2)  # (Z, Y)
    xz_proj_a = np.max(a_cropped, axis=1)  # (Z, X)

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
        "yx_proj_b": yx_proj_b,
        "zy_proj_b": zy_proj_b,
        "xz_proj_b": xz_proj_b,
        "yx_proj_a": yx_proj_a,
        "zy_proj_a": zy_proj_a,
        "xz_proj_a": xz_proj_a,
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

    Parameters
    ----------
    params : dict
        a_path : str - Remote path to View A zarr
        b_path : str - Remote path to View B zarr
        save_path : str - Remote output path (directory for OME-TIFF, file for zarr)
        output_format : str - "OME-TIFF" or "Zarr"
        psf_type : str - "gaussian" or "custom"
        psf_a_path : str | None (if custom)
        psf_b_path : str | None (if custom)
        deskew_method : str - "Orthogonal", "Shear-Warp", or "Affine"
            Determines theta angles for PSF rotation (Shear-Warp: 0/90 deg,
            others: 45/-45 deg).
        fwhm_a_lat : float - View A lateral FWHM in pixels.
        fwhm_a_ax : float - View A axial FWHM in pixels.
        fwhm_b_lat : float - View B lateral FWHM in pixels.
        fwhm_b_ax : float - View B axial FWHM in pixels.
        fwhm_x : float (legacy, fallback when per-view params absent)
        fwhm_y : float (legacy)
        fwhm_z : float (legacy)
        bp_type : str - "gaussian" or "custom" or "flipped_psf"
        bp_a_path : str | None (if custom)
        bp_b_path : str | None (if custom)
        bp_fwhm_a_lat : float - Backprojector A lateral FWHM.
        bp_fwhm_a_ax : float - Backprojector A axial FWHM.
        bp_fwhm_b_lat : float - Backprojector B lateral FWHM.
        bp_fwhm_b_ax : float - Backprojector B axial FWHM.
        decon_function : str - "additive", "efficient", or "dispim"
        num_iter : int
        epsilon : float
        req_both : bool
        boundary_correction : bool
        boundary_sigma : float
        chunkwise : bool
        chunk_size : list [z, y, x]
        overlap : list [z, y, x]

    Returns
    -------
    dict with keys: output_path, output_format
    """
    import os
    import tempfile

    import numpy as np
    import tifffile
    import zarr

    request_id = params.get("_request_id")
    stdout = sys.stdout

    def _progress(msg, pct=None):
        send_progress(stdout, request_id, msg, pct)

    # Extract parameters
    a_path = os.path.abspath(params["a_path"])
    b_path = os.path.abspath(params["b_path"])
    save_path = os.path.abspath(params["save_path"])
    logger.info(
        "[handle_deconvolve] a_path=%s, b_path=%s, save_path=%s",
        a_path, b_path, save_path,
    )
    output_format = params.get("output_format", "OME-TIFF")
    psf_type = params.get("psf_type", "gaussian").lower()
    deskew_method = params.get("deskew_method", "Orthogonal")
    # Per-view FWHM parameters
    fwhm_a_lat = params.get("fwhm_a_lat", 2.0)
    fwhm_a_ax = params.get("fwhm_a_ax", 7.0)
    fwhm_b_lat = params.get("fwhm_b_lat", 2.0)
    fwhm_b_ax = params.get("fwhm_b_ax", 7.0)
    # Backprojector FWHM parameters
    bp_fwhm_a_lat = params.get("bp_fwhm_a_lat", fwhm_a_lat)
    bp_fwhm_a_ax = params.get("bp_fwhm_a_ax", fwhm_a_ax)
    bp_fwhm_b_lat = params.get("bp_fwhm_b_lat", fwhm_b_lat)
    bp_fwhm_b_ax = params.get("bp_fwhm_b_ax", fwhm_b_ax)
    # Legacy fallback: if per-view params not provided, use old fwhm_x/y/z
    if "fwhm_a_lat" not in params and "fwhm_x" in params:
        fwhm_a_lat = params.get("fwhm_x", 2.0)
        fwhm_a_ax = params.get("fwhm_z", 7.0)
        fwhm_b_lat = fwhm_a_lat
        fwhm_b_ax = fwhm_a_ax
    bp_type = params.get("bp_type", "flipped_psf")
    decon_function = params.get("decon_function", "additive")
    num_iter = params.get("num_iter", 20)
    epsilon = params.get("epsilon", 0.001)
    req_both = params.get("req_both", True)
    boundary_correction = params.get("boundary_correction", False)
    boundary_sigma = params.get("boundary_sigma", 3.0)
    chunkwise = params.get("chunkwise", False)
    chunk_size = tuple(params.get("chunk_size", [400, 400, 400]))
    overlap = tuple(params.get("overlap", [100, 100, 100]))

    _progress("Loading input volumes...", 5)

    # Open input zarr arrays
    zarr_a = zarr.open(a_path, mode="r")
    zarr_b = zarr.open(b_path, mode="r")

    _progress("Generating PSF...", 10)

    # --- Compute theta angles from deskew method ---
    # Matches the logic in DeconvolutionWidget._get_psf_and_backprojectors()
    deskew_method_lower = deskew_method.lower().replace("-", "").replace(" ", "")
    if deskew_method_lower == "shearwarp" or deskew_method_lower == "shear":
        theta_a_deg = 0.0
        theta_b_deg = 90.0
    else:  # orthogonal, affine, dispim
        theta_a_deg = 45.0
        theta_b_deg = -45.0

    # --- Helper: generate Gaussian PSF with theta rotation ---
    def _make_gaussian_psf(fwhm_lat, fwhm_ax, theta_deg):
        """Generate a Gaussian PSF volume with optional ZX-plane rotation."""
        import math
        from scipy import ndimage
        from napari_pyspim._psf import generate_psf_im, normalize_psf_im

        fwhm_to_sigma = 1.0 / (2 * math.sqrt(2 * math.log(2)))
        sigma_lat = fwhm_lat * fwhm_to_sigma
        sigma_ax = fwhm_ax * fwhm_to_sigma

        # Calculate PSF size: FWHM * 3, rounded up to odd
        size = math.ceil(max(fwhm_ax, fwhm_lat) * 3)
        if size % 2 == 0:
            size += 1

        im_shape = (size, size, size)
        # pars: [x0, y0, z0, sx, sy, sz, ampl, bkgrnd]
        pars = [size / 2, size / 2, size / 2, sigma_lat, sigma_lat, sigma_ax, 1.0, 0.0]
        psf_im = generate_psf_im(pars, im_shape, "spherical")

        # Apply rotation in the ZX plane (axes 0 and 2) if theta is non-zero
        if theta_deg != 0.0:
            psf_im = ndimage.rotate(
                psf_im, theta_deg, axes=(0, 2), reshape=True,
                mode="constant", cval=0
            )

        return normalize_psf_im(psf_im)

    if psf_type == "gaussian":
        psf_a = _make_gaussian_psf(fwhm_a_lat, fwhm_a_ax, theta_a_deg)
        psf_b = _make_gaussian_psf(fwhm_b_lat, fwhm_b_ax, theta_b_deg)
    else:
        # Custom PSF: load from files
        psf_a_path = params.get("psf_a_path")
        psf_b_path = params.get("psf_b_path")
        if psf_a_path and psf_b_path:
            psf_a = _load_psf_file(psf_a_path)
            psf_b = _load_psf_file(psf_b_path)
        else:
            raise ValueError("Custom PSF selected but paths not provided")

    # --- Generate or load backprojectors ---
    if bp_type == "flipped_psf":
        bp_a = np.flip(psf_a).copy()
        bp_b = np.flip(psf_b).copy()
    elif bp_type == "gaussian":
        bp_a = _make_gaussian_psf(bp_fwhm_a_lat, bp_fwhm_a_ax, theta_a_deg)
        bp_b = _make_gaussian_psf(bp_fwhm_b_lat, bp_fwhm_b_ax, theta_b_deg)
    else:
        # Custom backprojector: load from files
        bp_a_path = params.get("bp_a_path")
        bp_b_path = params.get("bp_b_path")
        if bp_a_path and bp_b_path:
            bp_a = _load_psf_file(bp_a_path)
            bp_b = _load_psf_file(bp_b_path)
        else:
            raise ValueError("Custom backprojector selected but paths not provided")

    _progress("Starting deconvolution...", 15)

    # Import deconvolution functions
    from pyspim.decon.rl.dualview_fft import deconvolve, deconvolve_chunkwise

    if chunkwise:
        _progress("Running chunkwise deconvolution...", 20)
        # Use temporary zarr for chunkwise output
        with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_dir:
            out = zarr.open_array(
                tmp_dir, mode="w", shape=zarr_a.shape, dtype=np.float32, fill_value=0
            )

            # Compute overlap from PSF size
            psf_overlap = max(s // 2 for s in psf_a.shape)
            overlap = tuple([max(o, psf_overlap) for o in overlap])
            deconvolve_chunkwise(
                zarr_a,
                zarr_b,
                out,
                chunk_size,
                overlap,
                psf_a,
                psf_b,
                bp_a,
                bp_b,
                decon_function,
                num_iter,
                epsilon,
                boundary_correction,
                None,  # zero_padding
                boundary_sigma,
                boundary_sigma,
                verbose=True,
            )

            _progress("Saving output...", 90)
            output_path = _save_decon_result(out, save_path, output_format, zarr_a.shape)
    else:
        _progress("Running full deconvolution...", 20)
        import cupy

        # Convert PSFs and backprojectors to cupy
        psf_a_cp = cupy.asarray(psf_a, dtype=cupy.float32)
        psf_b_cp = cupy.asarray(psf_b, dtype=cupy.float32)
        bp_a_cp = cupy.asarray(bp_a, dtype=cupy.float32)
        bp_b_cp = cupy.asarray(bp_b, dtype=cupy.float32)

        # Load volumes into GPU memory
        a_data = cupy.asarray(np.asarray(zarr_a[:]), dtype=cupy.float32)
        b_data = cupy.asarray(np.asarray(zarr_b[:]), dtype=cupy.float32)

        result_cp = deconvolve(
            a_data,
            b_data,
            None,  # init
            psf_a_cp,
            psf_b_cp,
            bp_a_cp,
            bp_b_cp,
            decon_function,
            num_iter,
            epsilon,
            req_both,
            boundary_correction,
            None,  # zero_padding
            boundary_sigma,
            boundary_sigma,
            verbose=True,
        )

        result = result_cp.get().astype(np.float32)

        _progress("Saving output...", 90)
        output_path = _save_decon_result_array(result, save_path, output_format, zarr_a.shape)

    _progress("Deconvolution completed!", 100)

    return {"output_path": output_path, "output_format": output_format}


def _load_psf_file(path: str) -> "np.ndarray":
    """Load a PSF volume from .npy or .tif file."""
    import numpy as np
    import tifffile

    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    else:
        # Try numpy first, then tifffile
        try:
            return np.load(path)
        except Exception:
            return tifffile.imread(path)


def _save_decon_result(out: "zarr.Array", save_path: str, output_format: str, shape: tuple) -> str:
    """Save deconvolution result from a zarr array to the specified format.

    For Zarr output, copies the zarr folder to the destination.
    For OME-TIFF output, uses dask.array for lazy loading and tifffile
    for memory-efficient tiled writes.
    """
    import os
    import shutil

    is_multichannel = len(shape) > 3
    axes = "CZYX" if is_multichannel else "ZYX"

    if output_format == "Zarr":
        # Ensure path ends with .zarr
        if not save_path.endswith(".zarr"):
            save_path = save_path + ".zarr"

        # Copy the zarr directory to the final location.
        # Use the store's root directory if available, otherwise fall back
        # to copying chunk-by-chunk.
        source_path = getattr(out.store, "root", None)
        if source_path and os.path.isdir(source_path):
            # FilesystemStore: copy the entire directory
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            shutil.copytree(source_path, save_path)
        else:
            # Non-filesystem store: create target and copy chunk-by-chunk
            import zarr

            final_zarr = zarr.open(
                save_path, mode="w",
                shape=out.shape,
                dtype=out.dtype,
                chunks=out.chunks,
                compressor=out.compressor,
            )
            final_zarr[:] = out[:]

        return save_path
    else:
        # OME-TIFF — use dask.array for lazy loading + tifffile for tiled write
        if not save_path.endswith(".tif"):
            if not save_path.endswith(".tiff"):
                save_path = save_path + ".ome.tif"

        import dask.array as da
        import tifffile

        # Open the zarr array lazily (does not load into RAM)
        dask_data = da.from_zarr(out)

        # Write to OME-TIFF using tiled/chunked approach
        tifffile.imwrite(
            save_path,
            dask_data,
            bigtiff=True,
            photometric="minisblack",
            resolution=(1 / 0.1625, 1 / 0.1625),
            metadata={"axes": axes, "spacing": 0.1625, "units": "um"},
            tile=(1024, 1024),
        )
        return save_path


def _save_decon_result_array(result: "np.ndarray", save_path: str, output_format: str, shape: tuple) -> str:
    """Save deconvolution result from a numpy array to the specified format."""
    import zarr

    is_multichannel = len(shape) > 3
    axes = "CZYX" if is_multichannel else "ZYX"

    if output_format == "Zarr":
        if not save_path.endswith(".zarr"):
            save_path = save_path + ".zarr"
        final_zarr = zarr.open(
            save_path, mode="w",
            shape=result.shape,
            dtype=result.dtype,
        )
        final_zarr[:] = result
        return save_path
    else:
        # OME-TIFF
        if not save_path.endswith(".tif"):
            if not save_path.endswith(".tiff"):
                save_path = save_path + ".ome.tif"

        import tifffile

        tifffile.imwrite(
            save_path,
            result,
            bigtiff=True,
            photometric="minisblack",
            resolution=(1 / 0.1625, 1 / 0.1625),
            metadata={"axes": axes, "spacing": 0.1625, "units": "um"},
        )
        return save_path


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
            (chan_min, chan_max) 0-indexed inclusive range of channels.
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
    import tifffile
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
    save_tiffs = params.get("save_tiffs", False)
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
    camera_offset = dp.get("camera_offset", 0)
    theta_deg = 45.0  # Hardcoded to 45 degrees
    theta_rad = math.pi / 4
    affine_matrix = np.array(saved_params["affine_registration_matrix"])

    rp = saved_params["registration_parameters"]
    interp_method = rp.get("interp_method", "cubspl")
    crop_bounds = saved_params.get("crop_bounds")
    pre_reg = saved_params["pre_reg_transform"]
    pre_reg_t = [
        pre_reg["tz_um"] / pixel_size,
        pre_reg["ty_um"] / pixel_size,
        pre_reg["tx_um"] / pixel_size,
    ]

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

    # Determine channel indices
    chan_start = channel_range[0]
    chan_end = channel_range[1]
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

        # Subtract camera offset if non-zero
        if camera_offset > 0:
            from pyspim.data.dispim import subtract_constant_uint16arr
            vol_a = subtract_constant_uint16arr(vol_a, camera_offset)
            vol_b = subtract_constant_uint16arr(vol_b, camera_offset)

        a_dsk = dsk.deskew_stage_scan(
            vol_a, pixel_size, step_size, 1,
            theta=theta_rad, method=method, **deskew_kwargs,
        )
        try:
            a_dsk = a_dsk.get()
        except Exception:
            pass

        b_dsk = dsk.deskew_stage_scan(
            vol_b, pixel_size, step_size, -1,
            theta=theta_rad, method=method, **deskew_kwargs,
        )
        try:
            b_dsk = b_dsk.get()
        except Exception:
            pass

        if crop_bounds:
            z_start = crop_bounds["z_start"]
            z_end = crop_bounds["z_end"]
            y_start = crop_bounds["y_start"]
            y_end = crop_bounds["y_end"]
            x_start = crop_bounds["x_start"]
            x_end = crop_bounds["x_end"]
            tz, ty, tx = pre_reg_t
            a_dsk = a_dsk[z_start:z_end, y_start:y_end, x_start:x_end]
            Zb, Yb, Xb = b_dsk.shape
            z_start_b = max(0, int(z_start - tz))
            z_end_b = min(Zb, int(z_end - tz))
            y_start_b = max(0, int(y_start - ty))
            y_end_b = min(Yb, int(y_end - ty))
            x_start_b = max(0, int(x_start - tx))
            x_end_b = min(Xb, int(x_end - tx))
            b_dsk = b_dsk[z_start_b:z_end_b, y_start_b:y_end_b, x_start_b:x_end_b]

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
            f"Time {t}, Channel {first_chan} done ({percentage}%)", percentage,
        )

        # Helper to save TIFF after all channels are written
        def _save_tiff_if_needed(zarr_path, data_arr):
            from dask.array import from_zarr
            dask_data = from_zarr(data_arr)
            if save_tiffs:
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
            c = chan_idx - chan_start

            with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
                if multi_pos:
                    vol_a = acq.get(position, "a", chan_idx, t, window=window)
                    vol_b = acq.get(position, "b", chan_idx, t, window=window)
                else:
                    vol_a = acq.get("a", chan_idx, t, window=window)
                    vol_b = acq.get("b", chan_idx, t, window=window)

            # Subtract camera offset if non-zero
            if camera_offset > 0:
                from pyspim.data.dispim import subtract_constant_uint16arr
                vol_a = subtract_constant_uint16arr(vol_a, camera_offset)
                vol_b = subtract_constant_uint16arr(vol_b, camera_offset)

            a_dsk = dsk.deskew_stage_scan(
                vol_a, pixel_size, step_size, 1,
                theta=theta_rad, method=method, **deskew_kwargs,
            )
            try:
                a_dsk = a_dsk.get()
            except Exception:
                pass

            b_dsk = dsk.deskew_stage_scan(
                vol_b, pixel_size, step_size, -1,
                theta=theta_rad, method=method, **deskew_kwargs,
            )
            try:
                b_dsk = b_dsk.get()
            except Exception:
                pass

            if crop_bounds:
                z_start = crop_bounds["z_start"]
                z_end = crop_bounds["z_end"]
                y_start = crop_bounds["y_start"]
                y_end = crop_bounds["y_end"]
                x_start = crop_bounds["x_start"]
                x_end = crop_bounds["x_end"]
                tz, ty, tx = pre_reg_t
                a_dsk = a_dsk[z_start:z_end, y_start:y_end, x_start:x_end]
                Zb, Yb, Xb = b_dsk.shape
                z_start_b = max(0, int(z_start - tz))
                z_end_b = min(Zb, int(z_end - tz))
                y_start_b = max(0, int(y_start - ty))
                y_end_b = min(Yb, int(y_end - ty))
                x_start_b = max(0, int(x_start - tx))
                x_end_b = min(Xb, int(x_end - tx))
                b_dsk = b_dsk[z_start_b:z_end_b, y_start_b:y_end_b, x_start_b:x_end_b]

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
                f"Time {t}, Channel {chan_idx} done ({percentage}%)", percentage,
            )

        # Save TIFF files after all channels for this timepoint are written
        _save_tiff_if_needed(a_zarr_path, arr_a)
        _save_tiff_if_needed(b_zarr_path, arr_b)

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


# ---------------------------------------------------------------------------
# Batch job handlers
# ---------------------------------------------------------------------------

def handle_submit_batch_deconvolution(params: dict) -> dict:
    """Submit a deconvolution job via SLURM sbatch.

    Generates a batch script and params JSON, submits with sbatch,
    and returns the job_id and paths.
    """
    import subprocess

    batch_cfg = params.pop("batch_config", {})
    time_string = batch_cfg.get("time_string", "01:00:00")
    memory_gb = batch_cfg.get("memory_gb", 64)
    gpus = batch_cfg.get("gpus", 1)
    ntasks = batch_cfg.get("ntasks", 1)
    log_dir = batch_cfg.get("log_dir", "/tmp")

    # Get remote venv from environment or derive
    remote_venv = os.environ.get("VIRTUAL_ENV", "")
    if not remote_venv:
        # Try to find venv from sys.prefix
        remote_venv = sys.prefix

    # Use persistent logs directory on shared filesystem instead of /tmp
    root_dir = os.path.dirname(remote_venv)
    batch_dir = os.path.join(root_dir, "logs", "batch_jobs")
    os.makedirs(batch_dir, exist_ok=True)

    from napari_pyspim._batch_utils import (
        generate_batch_script,
        generate_params_json,
        get_unique_paths,
        get_batch_runner_path,
    )

    script_path, params_json_path, result_path = get_unique_paths(batch_dir, "deconvolution")
    batch_runner_path = get_batch_runner_path(remote_venv)

    # Write params JSON
    with open(params_json_path, "w") as f:
        f.write(generate_params_json("deconvolution", params))

    # Generate and write batch script
    script_content = generate_batch_script(
        command_type="deconvolution",
        params_json_path=params_json_path,
        result_path=result_path,
        remote_venv=remote_venv,
        batch_runner_path=batch_runner_path,
        time_string=time_string,
        memory_gb=memory_gb,
        gpus=gpus,
        ntasks=ntasks,
    )
    with open(script_path, "w") as f:
        f.write(script_content)

    # Submit
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    # Parse job ID from sbatch output: "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]

    return {
        "job_id": job_id,
        "result_path": result_path,
        "script_path": script_path,
        "log_dir": log_dir,
    }


def handle_submit_batch_registration(params: dict) -> dict:
    """Submit a registration job via SLURM sbatch.

    Writes deskewed volumes from session to zarr, generates batch script,
    submits with sbatch, and returns job_id and paths.
    """
    import subprocess

    batch_cfg = params.pop("batch_config", {})
    time_string = batch_cfg.get("time_string", "01:00:00")
    memory_gb = batch_cfg.get("memory_gb", 64)
    gpus = batch_cfg.get("gpus", 1)
    ntasks = batch_cfg.get("ntasks", 1)
    log_dir = batch_cfg.get("log_dir", "/tmp")

    remote_venv = os.environ.get("VIRTUAL_ENV", "") or sys.prefix

    # Use persistent logs directory on shared filesystem instead of /tmp
    root_dir = os.path.dirname(remote_venv)
    batch_dir = os.path.join(root_dir, "logs")
    os.makedirs(batch_dir, exist_ok=True)

    from napari_pyspim._batch_utils import (
        generate_batch_script,
        generate_params_json,
        get_unique_paths,
        get_batch_runner_path,
    )

    session_id = params.get("session_id")
    session = SESSIONS.get(session_id)
    if session is None:
        raise KeyError(f"Session '{session_id}' not found on server")

    # Write deskewed volumes to zarr in persistent directory
    import zarr
    script_path, params_json_path, result_path = get_unique_paths(batch_dir, "registration")

    a_zarr_path = params_json_path.replace("_params.json", "_a.zarr")
    b_zarr_path = params_json_path.replace("_params.json", "_b.zarr")

    zarr.open(a_zarr_path, mode="w", data=session["a_deskewed"])
    zarr.open(b_zarr_path, mode="w", data=session["b_deskewed"])

    # Build computation params with zarr paths
    reg_params = dict(params)
    reg_params["a_zarr_path"] = a_zarr_path
    reg_params["b_zarr_path"] = b_zarr_path
    reg_params.pop("session_id", None)

    # Write params JSON
    with open(params_json_path, "w") as f:
        f.write(generate_params_json("registration", reg_params))

    # Generate and write batch script
    batch_runner_path = get_batch_runner_path(remote_venv)
    script_content = generate_batch_script(
        command_type="registration",
        params_json_path=params_json_path,
        result_path=result_path,
        remote_venv=remote_venv,
        batch_runner_path=batch_runner_path,
        time_string=time_string,
        memory_gb=memory_gb,
        gpus=gpus,
        ntasks=ntasks,
    )
    with open(script_path, "w") as f:
        f.write(script_content)

    # Submit
    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr}")

    job_id = result.stdout.strip().split()[-1]

    return {
        "job_id": job_id,
        "result_path": result_path,
        "script_path": script_path,
        "log_dir": log_dir,
    }


def handle_check_job_status(params: dict) -> dict:
    """Check the status of a SLURM job using squeue."""
    import subprocess

    job_id = params["job_id"]
    try:
        result = subprocess.run(
            ["squeue", "--job", job_id, "--noheader", "--format", "%T"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            # squeue returns non-zero if job not found
            return {"status": "NOT_FOUND"}
        status = result.stdout.strip()
        return {"status": status}
    except subprocess.TimeoutExpired:
        return {"status": "NOT_FOUND"}
    except FileNotFoundError:
        return {"status": "NOT_FOUND", "error": "squeue not found"}


def handle_read_batch_results(params: dict) -> dict:
    """Read batch job results from JSON and move logs to log_dir."""
    import shutil

    result_path = params["result_path"]
    log_dir = params.get("log_dir", "")

    # Read results
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")

    with open(result_path) as f:
        result_data = json.load(f)

    # Move batch files to log_dir if specified
    if log_dir and os.path.isdir(log_dir):
        _move_batch_logs_to_logdir(result_path, log_dir)

    return result_data


def _move_batch_logs_to_logdir(result_path: str, log_dir: str):
    """Move batch script and log files to the designated log directory."""
    import glob
    import re

    # Extract base prefix from result_path: /tmp/pyspim_batch_deconvolution_abcdef_results.json
    base = result_path.replace("_results.json", "")
    # Find related files: .sh, .out, .err
    pattern = re.compile(re.escape(base) + r"[\w_-]*\.(sh|out|err)$")
    tmp_dir = os.path.dirname(base)

    for filepath in glob.glob(os.path.join(tmp_dir, "pyspim_batch_*")):
        if pattern.match(filepath):
            dest = os.path.join(log_dir, os.path.basename(filepath))
            try:
                shutil.move(filepath, dest)
            except (shutil.Error, OSError):
                pass  # Best effort

    # Also move the results JSON
    dest = os.path.join(log_dir, os.path.basename(result_path))
    try:
        shutil.move(result_path, dest)
    except (shutil.Error, OSError):
        pass


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
    "submit_batch_deconvolution": handle_submit_batch_deconvolution,
    "submit_batch_registration": handle_submit_batch_registration,
    "check_job_status": handle_check_job_status,
    "read_batch_results": handle_read_batch_results,
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
