"""
Remote computation server for napari-pyspim.

Reads MessagePack commands from stdin, executes pyspim operations,
and writes MessagePack responses to stdout.

This script is designed to run on a remote server, activated via SSH.
It maintains state (temp directories, etc.) for the duration of the session.
"""

import json
import math
import os
import sys
import tempfile
import traceback

import msgpack
import numpy as np

# Track temporary zarr directories for cleanup on exit
_temp_dirs = []


def encode_array(obj):
    """Encode numpy arrays for MessagePack serialization."""
    if isinstance(obj, np.ndarray):
        return msgpack.ExtType(0, msgpack.packb({
            'dtype': str(obj.dtype),
            'shape': obj.shape,
            'data': obj.tobytes(),
        }))
    raise TypeError(f"Cannot encode object of type {type(obj)}")


def decode_array(code, data, header):
    """Decode numpy arrays from MessagePack ext type."""
    if code == 0:
        meta = msgpack.unpackb(data, raw=False)
        arr = np.frombuffer(meta['data'], dtype=np.dtype(meta['dtype']))
        return arr.reshape(meta['shape'])
    return msgpack.ExtType(code, data)


def send_response(request_id, result, error=None, progress=None):
    """Serialize and send a response to stdout."""
    msg = {
        'request_id': request_id,
        'success': error is None,
        'result': result,
        'error': error,
        'progress': progress,
    }
    try:
        packed = msgpack.packb(msg, default=encode_array)
        # Send length-prefixed message
        length = len(packed)
        sys.stdout.buffer.write(length.to_bytes(8, 'big'))
        sys.stdout.buffer.write(packed)
        sys.stdout.buffer.flush()
    except Exception as e:
        sys.stderr.write(f"Error sending response: {e}\n")
        sys.stderr.flush()


def receive_message():
    """Read a length-prefixed MessagePack message from stdin."""
    header = _read_exact(sys.stdin.buffer, 8)
    if len(header) < 8:
        return None  # EOF
    length = int.from_bytes(header, 'big')
    data = _read_exact(sys.stdin.buffer, length)
    return msgpack.unpackb(data, ext_hook=decode_array, raw=False)


def _read_exact(stream, n):
    """Read exactly n bytes from a stream."""
    data = b''
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            break
        data += chunk
    return data


def cleanup_temp_dirs():
    """Remove all tracked temporary directories."""
    import shutil
    for d in _temp_dirs:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
        except Exception as e:
            sys.stderr.write(f"Error cleaning up {d}: {e}\n")
            sys.stderr.flush()


# --- Command Handlers ---

def handle_ping(params, **kwargs):
    """Health check."""
    import pyspim
    return {
        'status': 'ok',
        'pyspim_version': getattr(pyspim, '__version__', 'unknown'),
        'numpy_version': np.__version__,
    }


def handle_compute_projections(params, request_id=None):
    """Load data and compute projections for ROI detection."""
    from pyspim.data import dispim as data

    data_path = params['data_path']
    channel = params['channel']
    projection_type = params['projection_type']
    multi_pos = params.get('multi_pos', False)
    time = params.get('time', 0)
    position = params.get('position', 0)

    send_response(request_id, None, progress="Server: loading data...")

    with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
        if multi_pos:
            volume_a = acq.get(position, 'a', channel, time)
            volume_b = acq.get(position, 'b', channel, time)
        else:
            volume_a = acq.get('a', channel, time)
            volume_b = acq.get('b', channel, time)

    send_response(request_id, None, progress="Server: data loaded, computing projections...")

    def compute_projs(vol, proj_type):
        if proj_type == 'max':
            yx = np.max(vol, axis=0)
            zy = np.max(vol, axis=2)
        elif proj_type == 'sum':
            yx = np.sum(vol, axis=0)
            zy = np.sum(vol, axis=2)
        elif proj_type == 'mean':
            yx = np.mean(vol, axis=0)
            zy = np.mean(vol, axis=2)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        return yx, zy

    yx_proj_a, zy_proj_a = compute_projs(volume_a, projection_type)
    yx_proj_b, zy_proj_b = compute_projs(volume_b, projection_type)

    send_response(request_id, None, progress="Server: projections computed, sending result...")

    return {
        'yx_proj_a': yx_proj_a,
        'zy_proj_a': zy_proj_a,
        'yx_proj_b': yx_proj_b,
        'zy_proj_b': zy_proj_b,
        'volume_shape_a': list(volume_a.shape),
        'volume_shape_b': list(volume_b.shape),
        'data_path': data_path,
    }


def handle_load_deskew(params, **kwargs):
    """Load data, deskew, compute projections, and save volumes as zarr."""
    import zarr
    from pyspim.data import dispim as data
    from pyspim import deskew as dsk

    data_path = params['data_path']
    channel = params['channel']
    projection_type = params['projection_type']
    pixel_size = params['pixel_size']
    step_size = params['step_size']
    theta = params['theta']
    method = params['method']
    ignore_bbox = params.get('ignore_bbox', False)
    multi_pos = params.get('multi_pos', False)
    time = params.get('time', 0)
    position = params.get('position', 0)
    auto_crop = params.get('auto_crop', True)

    # Load bounding box if present
    window = None
    if not ignore_bbox:
        bbox_path = os.path.join(data_path, 'bbox_raw.json')
        if os.path.exists(bbox_path):
            try:
                with open(bbox_path, 'r') as f:
                    bbox = json.load(f)
                window = (
                    slice(bbox[0][0], bbox[0][1]),
                    slice(bbox[1][0], bbox[1][1]),
                    slice(bbox[2][0], bbox[2][1]),
                )
            except (json.JSONDecodeError, IndexError, TypeError):
                pass

    # Load data
    with data.uManagerAcquisition(data_path, multi_pos, np) as acq:
        if multi_pos:
            volume_a = acq.get(position, 'a', channel, time, window=window)
            volume_b = acq.get(position, 'b', channel, time, window=window)
        else:
            volume_a = acq.get('a', channel, time, window=window)
            volume_b = acq.get('b', channel, time, window=window)

    # Calculate lateral step size
    step_size_lat = step_size / math.cos(theta)

    # Deskew kwargs
    if method == 'shear':
        deskew_kwargs_a = {
            'rotation_thetas': (0, 0, 0),
            'interp_method': 'linear',
            'auto_crop': auto_crop,
            'preserve_dtype': True,
            'block_size': (8, 8, 8),
        }
        deskew_kwargs_b = dict(deskew_kwargs_a)
        deskew_kwargs_b['rotation_thetas'] = (0, math.pi / 2, 0)
    elif method == 'ortho':
        deskew_kwargs_a = {'preserve_dtype': True, 'stream': None}
        deskew_kwargs_b = dict(deskew_kwargs_a)
    else:  # dispim
        deskew_kwargs_a = {'preserve_dtype': True}
        deskew_kwargs_b = {'preserve_dtype': True}

    # Deskew View A
    a_dsk = dsk.deskew_stage_scan(
        volume_a, pixel_size, step_size_lat, 1,
        theta=theta, method=method, **deskew_kwargs_a,
    )
    try:
        a_dsk = a_dsk.get()
    except AttributeError:
        pass

    # Deskew View B
    b_dsk = dsk.deskew_stage_scan(
        volume_b, pixel_size, step_size_lat, -1,
        theta=theta, method=method, **deskew_kwargs_b,
    )
    if method == 'dispim':
        from pyspim.interp.affine import transform as affine_transform
        from pyspim._matrix import rotation_about_point_matrix
        R = rotation_about_point_matrix(
            0, math.pi / 2, 0,
            *[s / 2 for s in b_dsk.shape[::-1]]
        )
        b_dsk = affine_transform(
            b_dsk, R, 'linear', True,
            (b_dsk.shape[0], b_dsk.shape[1], b_dsk.shape[2]),
            8, 8, 8
        )
    try:
        b_dsk = b_dsk.get()
    except AttributeError:
        pass

    # Compute projections
    def compute_projs(vol, proj_type):
        if proj_type == 'max':
            yx = np.max(vol, axis=0)
            zy = np.max(vol, axis=2)
            xz = np.max(vol, axis=1)
        elif proj_type == 'sum':
            yx = np.sum(vol, axis=0)
            zy = np.sum(vol, axis=2)
            xz = np.sum(vol, axis=1)
        elif proj_type == 'mean':
            yx = np.mean(vol, axis=0)
            zy = np.mean(vol, axis=2)
            xz = np.mean(vol, axis=1)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        return yx, zy, xz

    yx_proj_a, zy_proj_a, xz_proj_a = compute_projs(a_dsk, projection_type)
    yx_proj_b, zy_proj_b, xz_proj_b = compute_projs(b_dsk, projection_type)

    # Save deskewed volumes as zarr in temp directory
    temp_dir = tempfile.mkdtemp(prefix='pyspim_dsk_')
    _temp_dirs.append(temp_dir)

    a_zarr_path = os.path.join(temp_dir, 'a_deskewed.zarr')
    b_zarr_path = os.path.join(temp_dir, 'b_deskewed.zarr')

    zarr.open(a_zarr_path, mode='w', shape=a_dsk.shape,
              dtype=a_dsk.dtype, chunks=(64, 256, 256))[:] = a_dsk
    zarr.open(b_zarr_path, mode='w', shape=b_dsk.shape,
              dtype=b_dsk.dtype, chunks=(64, 256, 256))[:] = b_dsk

    return {
        'a_zarr_path': a_zarr_path,
        'b_zarr_path': b_zarr_path,
        'yx_proj_a': yx_proj_a,
        'zy_proj_a': zy_proj_a,
        'xz_proj_a': xz_proj_a,
        'yx_proj_b': yx_proj_b,
        'zy_proj_b': zy_proj_b,
        'xz_proj_b': xz_proj_b,
        'volume_shape_a': list(a_dsk.shape),
        'volume_shape_b': list(b_dsk.shape),
        'method': method,
        'step_size_lat': step_size_lat,
        'temp_dir': temp_dir,
    }


def handle_register(params, **kwargs):
    """Register two deskewed volumes."""
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp
    import zarr
    from pyspim.interp import affine
    from pyspim.reg import powell
    from pyspim.util import launch_params_for_volume, pad_to_same_size

    # Read input volumes from remote zarr paths
    a_zarr_path = params['a_zarr_path']
    b_zarr_path = params['b_zarr_path']
    a_dsk = np.asarray(zarr.open(a_zarr_path, mode='r')[:])
    b_dsk = np.asarray(zarr.open(b_zarr_path, mode='r')[:])

    transform_type = params['transform_type']
    initial_translation = params.get('initial_translation', [0, 0, 0])
    metric = params.get('metric', 'ncc')
    interp_method = params.get('interp_method', 'cubspl')
    use_piecewise = params.get('use_piecewise', True)
    bound_translation = params.get('bound_translation', 20.0)
    bound_rot_shear = params.get('bound_rot_shear', 5.0)
    bound_scale = params.get('bound_scale', 0.05)

    shp_a = a_dsk.shape
    shp_b = b_dsk.shape

    # Pad volumes to same size if needed
    if shp_a[0] != shp_b[0] or shp_a[1] != shp_b[1] or shp_a[2] != shp_b[2]:
        a_dsk, b_dsk = pad_to_same_size(a_dsk, b_dsk)

    # Set up initial parameters and bounds
    t0 = initial_translation
    bt = bound_translation
    br = bound_rot_shear
    bs = bound_scale

    if transform_type == 't':
        par0 = t0
        bounds = [(t - bt, t + bt) for t in t0]
    elif transform_type == 't+r':
        par0 = np.concatenate([t0, np.asarray([0, 0, 0])])
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == 't+r+s':
        par0 = np.concatenate([t0, np.asarray([0, 0, 0]), np.asarray([1, 1, 1])])
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3 + [(1 - bs, 1 + bs)] * 3
    elif transform_type == 't+sh':
        par0 = np.concatenate([t0, np.asarray([0, 0, 0, 0, 0, 0])])
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6
    elif transform_type == 't+ssh':
        par0 = np.concatenate([t0, np.asarray([0, 0, 0])])
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == 't+sh+s':
        par0 = np.concatenate([t0, np.asarray([0, 0, 0, 0, 0, 0]), np.asarray([1, 1, 1])])
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6 + [(1 - bs, 1 + bs)] * 3

    # Launch parameters for GPU
    launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)

    # Perform optimization
    if use_piecewise:
        T, res = powell.optimize_affine_piecewise(
            cp.asarray(a_dsk), cp.asarray(b_dsk),
            metric=metric, transform=transform_type,
            interp_method=interp_method, par0=par0,
            bounds=bounds, kernel_launch_params=launch_par,
            verbose=False,
        )
    else:
        T, res = powell.optimize_affine(
            cp.asarray(a_dsk), cp.asarray(b_dsk),
            metric=metric, transform=transform_type,
            interp_method=interp_method, par0=par0,
            bounds=bounds, kernel_launch_params=launch_par,
            verbose=False,
        )

    # Apply transformation
    b_reg = affine.transform(
        cp.asarray(b_dsk), T,
        interp_method=interp_method, preserve_dtype=True,
        out_shp=None, block_size_z=8, block_size_y=8, block_size_x=8,
    ).get()

    # Crop to smallest size
    min_sze = [min(a, b) for a, b in zip(a_dsk.shape, b_reg.shape)]
    a_final = a_dsk[:min_sze[0], :min_sze[1], :min_sze[2]]
    b_final = b_reg[:min_sze[0], :min_sze[1], :min_sze[2]]

    # Calculate correlation ratio
    cr = 1 - res.fun

    # Save registered volumes as zarr in temp directory
    temp_dir = tempfile.mkdtemp(prefix='pyspim_reg_')
    _temp_dirs.append(temp_dir)

    a_zarr_path = os.path.join(temp_dir, 'a_registered.zarr')
    b_zarr_path = os.path.join(temp_dir, 'b_registered.zarr')

    zarr.open(a_zarr_path, mode='w', shape=a_final.shape,
              dtype=a_final.dtype, chunks=(64, 256, 256))[:] = a_final
    zarr.open(b_zarr_path, mode='w', shape=b_final.shape,
              dtype=b_final.dtype, chunks=(64, 256, 256))[:] = b_final

    return {
        'a_zarr_path': a_zarr_path,
        'b_zarr_path': b_zarr_path,
        'transform_matrix': T,
        'correlation_ratio': cr,
        'transform_type': transform_type,
        'shape': list(a_final.shape),
        'dtype': str(a_final.dtype),
        'temp_dir': temp_dir,
    }


def handle_deconvolve(params, **kwargs):
    """Perform dual-view deconvolution."""
    try:
        import cupy
    except ImportError:
        cupy = None
    import zarr
    from pyspim.decon.rl.dualview_fft import deconvolve, deconvolve_chunkwise

    # Get input views - either from zarr paths or from arrays
    if 'view_a_zarr_path' in params:
        view_a = np.asarray(zarr.open(params['view_a_zarr_path'], mode='r')[:])
        view_b = np.asarray(zarr.open(params['view_b_zarr_path'], mode='r')[:])
    else:
        view_a = params['view_a']
        view_b = params['view_b']

    psf_a = params['psf_a']
    psf_b = params['psf_b']
    backproj_a = params['backproj_a']
    backproj_b = params['backproj_b']
    decon_function = params['decon_function']
    num_iter = params['num_iter']
    epsilon = params['epsilon']
    req_both = params.get('req_both', True)
    boundary_correction = params.get('boundary_correction', False)
    boundary_sigma = params.get('boundary_sigma', 0.01)
    chunkwise = params.get('chunkwise', False)
    chunk_size = tuple(params.get('chunk_size', (64, 128, 128)))
    overlap = tuple(params.get('overlap', (40, 40, 40)))

    if chunkwise:
        with tempfile.TemporaryDirectory(suffix='.zarr') as tmp_dir:
            shape = view_a.shape
            dtype = np.float32

            tmp_a = os.path.join(tmp_dir, 'view_a.zarr')
            tmp_b = os.path.join(tmp_dir, 'view_b.zarr')
            tmp_out = os.path.join(tmp_dir, 'output.zarr')

            zarr_a = zarr.open(tmp_a, mode='w', shape=shape, dtype=dtype)
            zarr_b = zarr.open(tmp_b, mode='w', shape=shape, dtype=dtype)
            zarr_out = zarr.open(tmp_out, mode='w', shape=shape, dtype=dtype, fill_value=0)

            zarr_a[:] = view_a
            zarr_b[:] = view_b

            deconvolve_chunkwise(
                view_a=zarr_a, view_b=zarr_b, out=zarr_out,
                chunk_size=chunk_size, overlap=overlap,
                psf_a=np.asarray(psf_a, dtype=np.float32),
                psf_b=np.asarray(psf_b, dtype=np.float32),
                bp_a=np.asarray(backproj_a, dtype=np.float32),
                bp_b=np.asarray(backproj_b, dtype=np.float32),
                decon_function=decon_function,
                num_iter=num_iter, epsilon=epsilon,
                boundary_correction=boundary_correction,
                zero_padding=None,
                boundary_sigma_a=boundary_sigma,
                boundary_sigma_b=boundary_sigma,
                verbose=True,
            )

            result = np.asarray(zarr_out[:]).astype(np.float32)
    else:
        view_a_gpu = cupy.asarray(view_a, dtype=cupy.float32)
        view_b_gpu = cupy.asarray(view_b, dtype=cupy.float32)
        psf_a_gpu = cupy.asarray(psf_a, dtype=cupy.float32)
        psf_b_gpu = cupy.asarray(psf_b, dtype=cupy.float32)
        backproj_a_gpu = cupy.asarray(backproj_a, dtype=cupy.float32)
        backproj_b_gpu = cupy.asarray(backproj_b, dtype=cupy.float32)

        result_gpu = deconvolve(
            view_a=view_a_gpu, view_b=view_b_gpu, est_i=None,
            psf_a=psf_a_gpu, psf_b=psf_b_gpu,
            backproj_a=backproj_a_gpu, backproj_b=backproj_b_gpu,
            decon_function=decon_function,
            num_iter=num_iter, epsilon=epsilon,
            req_both=req_both,
            boundary_correction=boundary_correction,
            zero_padding=None,
            boundary_sigma_a=boundary_sigma,
            boundary_sigma_b=boundary_sigma,
            verbose=False,
        )
        result = result_gpu.get().astype(np.float32)

    # Save result as zarr in temp directory
    temp_dir = tempfile.mkdtemp(prefix='pyspim_decon_')
    _temp_dirs.append(temp_dir)

    result_zarr_path = os.path.join(temp_dir, 'deconvolved.zarr')
    zarr.open(result_zarr_path, mode='w', shape=result.shape,
              dtype=result.dtype, chunks=(64, 256, 256))[:] = result

    return {
        'result_zarr_path': result_zarr_path,
        'shape': list(result.shape),
        'dtype': str(result.dtype),
        'temp_dir': temp_dir,
    }


def handle_save_zarr(params, **kwargs):
    """Move zarr archive from temp to permanent location."""
    import shutil
    temp_path = params['temp_path']
    permanent_path = params['permanent_path']

    # Ensure parent directory exists
    parent = os.path.dirname(permanent_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    shutil.move(temp_path, permanent_path)

    # Remove from tracking if present
    if temp_path in _temp_dirs:
        _temp_dirs.remove(temp_path)

    return {'success': True, 'permanent_path': permanent_path}


def handle_cleanup_zarr(params, **kwargs):
    """Remove a temp zarr directory."""
    import shutil
    temp_path = params.get('temp_path')
    if temp_path and os.path.exists(temp_path):
        shutil.rmtree(temp_path)
        if temp_path in _temp_dirs:
            _temp_dirs.remove(temp_path)
    return {'success': True}


def handle_list_directory(params, **kwargs):
    """List directory contents via os.listdir."""
    remote_path = params['path']
    try:
        entries = []
        for name in os.listdir(remote_path):
            full_path = os.path.join(remote_path, name)
            entries.append({
                'name': name,
                'is_dir': os.path.isdir(full_path),
                'size': os.path.getsize(full_path) if os.path.isfile(full_path) else 0,
            })
        return {'entries': entries, 'path': remote_path}
    except Exception as e:
        return {'entries': [], 'path': remote_path, 'error': str(e)}


def handle_check_path(params, **kwargs):
    """Check if a path exists and is a directory."""
    path = params['path']
    return {
        'exists': os.path.exists(path),
        'is_dir': os.path.isdir(path),
        'is_file': os.path.isfile(path),
    }


# --- Command Dispatcher ---

COMMAND_HANDLERS = {
    'ping': handle_ping,
    'compute_projections': handle_compute_projections,
    'load_deskew': handle_load_deskew,
    'register': handle_register,
    'deconvolve': handle_deconvolve,
    'save_zarr': handle_save_zarr,
    'cleanup_zarr': handle_cleanup_zarr,
    'list_directory': handle_list_directory,
    'check_path': handle_check_path,
}


def main():
    """Main entry point for the remote server."""
    # Register cleanup on exit
    import atexit
    atexit.register(cleanup_temp_dirs)

    # Send ready message
    send_response(None, {'status': 'ready'})

    # Main loop
    while True:
        try:
            message = receive_message()
            if message is None:
                break  # EOF

            request_id = message.get('request_id')
            command = message.get('command')
            params = message.get('params', {})

            handler = COMMAND_HANDLERS.get(command)
            if handler is None:
                send_response(request_id, None, error=f"Unknown command: {command}")
                continue

            send_response(request_id, None, progress=f"Server: received '{command}' command")

            try:
                result = handler(params, request_id=request_id)
                send_response(request_id, result)
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()
                send_response(request_id, None, error=f"{type(e).__name__}: {e}")

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            break


if __name__ == '__main__':
    main()
