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


def handle_register(params: dict) -> dict:
    """Register two deskewed volumes.

    Mirrors ``RegistrationWorker.run()`` from ``_registration.py``.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("register not yet implemented")


def handle_deconvolve(params: dict) -> dict:
    """Dual-view Richardson-Lucy deconvolution.

    Mirrors ``DeconvolutionWorker.run()`` from ``_deconvolution.py``.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("deconvolve not yet implemented")


def handle_apply_registration(params: dict) -> dict:
    """Batch-apply deskew + registration across time/channel dimensions.

    TODO: Implement in Phase 4.
    """
    raise NotImplementedError("apply_registration not yet implemented")


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
        import pyspim  # type: ignore
        return getattr(pyspim, "__version__", "unknown")
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
            result = handler(params)
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
