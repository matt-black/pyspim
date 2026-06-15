#!/usr/bin/env python
"""Standalone runner for batch job computations.

Reads command and parameters from a JSON file, executes the computation,
and writes the result to an output JSON file.

Invoked by SLURM batch scripts on remote compute nodes.
"""

import argparse
import json
import os
import sys
import traceback


def _custom_encoder(obj):
    """Handle numpy arrays for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return {"__numpy__": True, "dtype": str(obj.dtype), "shape": list(obj.shape), "data": obj.tobytes().hex()}
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="pyspim batch job runner")
    parser.add_argument("--command", required=True, help="Command type (deconvolution or registration)")
    parser.add_argument("--params-json", required=True, help="Path to JSON params file")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    args = parser.parse_args()

    # Load parameters
    with open(args.params_json) as f:
        payload = json.load(f)

    command_type = payload.get("command_type", args.command)
    params = payload.get("params", {})

    output = {"success": False, "result": None, "error": None}

    try:
        if command_type == "deconvolution":
            result = _run_deconvolution(params)
            output["success"] = True
            output["result"] = result
        elif command_type == "registration":
            result = _run_registration(params)
            output["success"] = True
            output["result"] = result
        else:
            output["error"] = f"Unknown command type: {command_type}"

    except Exception as e:
        output["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    # Write results
    with open(args.output, "w") as f:
        json.dump(output, f, default=_custom_encoder, indent=2)


def _run_deconvolution(params: dict) -> dict:
    """Execute deconvolution using the remote server handler."""
    import numpy as np
    import zarr

    # Extract parameters
    a_path = os.path.abspath(params["a_path"])
    b_path = os.path.abspath(params["b_path"])
    save_path = os.path.abspath(params["save_path"])
    output_format = params.get("output_format", "OME-TIFF")
    psf_type = params.get("psf_type", "gaussian").lower()
    deskew_method = params.get("deskew_method", "Orthogonal")
    fwhm_a_lat = params.get("fwhm_a_lat", 2.0)
    fwhm_a_ax = params.get("fwhm_a_ax", 7.0)
    fwhm_b_lat = params.get("fwhm_b_lat", 2.0)
    fwhm_b_ax = params.get("fwhm_b_ax", 7.0)
    bp_fwhm_a_lat = params.get("bp_fwhm_a_lat", fwhm_a_lat)
    bp_fwhm_a_ax = params.get("bp_fwhm_a_ax", fwhm_a_ax)
    bp_fwhm_b_lat = params.get("bp_fwhm_b_lat", fwhm_b_lat)
    bp_fwhm_b_ax = params.get("bp_fwhm_b_ax", fwhm_b_ax)
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

    # Sparse RL parameters
    lambda1 = params.get("lambda1", 0.0)
    lambda2 = params.get("lambda2", 0.0)
    epsilon_hess = params.get("epsilon_hess", 1e-5)

    # Load input volumes
    zarr_a = zarr.open(a_path, mode="r")
    zarr_b = zarr.open(b_path, mode="r")

    # Generate PSFs
    psf_a, psf_b = _generate_psf_pair(psf_type, fwhm_a_lat, fwhm_a_ax, fwhm_b_lat, fwhm_b_ax,
                                        deskew_method, params)
    bp_a, bp_b = _generate_backprojector_pair(bp_type, psf_a, psf_b,
                                                bp_fwhm_a_lat, bp_fwhm_a_ax,
                                                bp_fwhm_b_lat, bp_fwhm_b_ax,
                                                deskew_method, params)

    if decon_function == "sparse_rl":
        from pyspim.decon.sparse.dualview_rl import deconvolve, deconvolve_chunkwise

        if chunkwise:
            import tempfile

            with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_dir:
                out = zarr.open_array(
                    tmp_dir, mode="w", shape=zarr_a.shape, dtype=np.float32, fill_value=0
                )
                deconvolve_chunkwise(
                    view_a=zarr_a,
                    view_b=zarr_b,
                    out=out,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    psf_a=psf_a,
                    psf_b=psf_b,
                    bp_a=bp_a,
                    bp_b=bp_b,
                    num_iter=num_iter,
                    epsilon=epsilon,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    epsilon_hess=epsilon_hess,
                    verbose=True,
                    decon_function="sparse",
                )
                output_path = _save_result(out, save_path, output_format, zarr_a.shape)
        else:
            import cupy

            a_data = cupy.asarray(np.asarray(zarr_a[:]), dtype=cupy.float32)
            b_data = cupy.asarray(np.asarray(zarr_b[:]), dtype=cupy.float32)
            psf_a_cp = cupy.asarray(psf_a, dtype=cupy.float32)
            psf_b_cp = cupy.asarray(psf_b, dtype=cupy.float32)
            bp_a_cp = cupy.asarray(bp_a, dtype=cupy.float32)
            bp_b_cp = cupy.asarray(bp_b, dtype=cupy.float32)

            result_cp = deconvolve(
                view_a=a_data,
                view_b=b_data,
                est_i=None,
                psf_a=psf_a_cp,
                psf_b=psf_b_cp,
                backproj_a=bp_a_cp,
                backproj_b=bp_b_cp,
                num_iter=num_iter,
                epsilon=epsilon,
                lambda1=lambda1,
                lambda2=lambda2,
                epsilon_hess=epsilon_hess,
                req_both=req_both,
                verbose=True,
            )
            result = result_cp.get().astype(np.float32)
            output_path = _save_result_array(result, save_path, output_format, zarr_a.shape)
    else:
        from pyspim.decon.rl.dualview_fft import deconvolve, deconvolve_chunkwise

        if chunkwise:
            import tempfile
            psf_overlap = max(s // 2 for s in psf_a.shape)
            overlap = tuple(max(o, psf_overlap) for o in overlap)

            with tempfile.TemporaryDirectory(suffix=".zarr") as tmp_dir:
                out = zarr.open_array(
                    tmp_dir, mode="w", shape=zarr_a.shape, dtype=np.float32, fill_value=0
                )
                deconvolve_chunkwise(
                    zarr_a, zarr_b, out, chunk_size, overlap,
                    psf_a, psf_b, bp_a, bp_b,
                    decon_function, num_iter, epsilon,
                    boundary_correction, None, boundary_sigma, boundary_sigma,
                    verbose=True,
                )
                output_path = _save_result(out, save_path, output_format, zarr_a.shape)
        else:
            import cupy
            a_data = cupy.asarray(np.asarray(zarr_a[:]), dtype=cupy.float32)
            b_data = cupy.asarray(np.asarray(zarr_b[:]), dtype=cupy.float32)
            psf_a_cp = cupy.asarray(psf_a, dtype=cupy.float32)
            psf_b_cp = cupy.asarray(psf_b, dtype=cupy.float32)
            bp_a_cp = cupy.asarray(bp_a, dtype=cupy.float32)
            bp_b_cp = cupy.asarray(bp_b, dtype=cupy.float32)

            result_cp = deconvolve(
                a_data, b_data, None,
                psf_a_cp, psf_b_cp, bp_a_cp, bp_b_cp,
                decon_function, num_iter, epsilon, req_both,
                boundary_correction, None, boundary_sigma, boundary_sigma,
                verbose=True,
            )
            result = result_cp.get().astype(np.float32)
            output_path = _save_result_array(result, save_path, output_format, zarr_a.shape)

    return {"output_path": output_path, "output_format": output_format}


def _run_registration(params: dict) -> dict:
    """Execute registration using parameters from a batch job."""
    import math
    import numpy as np

    try:
        import cupy as cp
    except ImportError:
        cp = np

    a_zarr_path = params["a_zarr_path"]
    b_zarr_path = params["b_zarr_path"]
    transform_type = params["transform_type"]
    pre_reg_translation = params.get("pre_reg_translation", [0, 0, 0])
    crop_bounds = params.get("crop_bounds")
    metric = params.get("metric", "cr")
    interp_method = params.get("interp_method", "cubspl")
    opt_method = params.get("opt_method", "powell")
    use_piecewise = params.get("use_piecewise", True)
    bound_translation = params.get("bound_translation", 20.0)
    bound_rot_shear = params.get("bound_rot_shear", 5.0)
    bound_scale = params.get("bound_scale", 0.05)

    # Load deskewed volumes from zarr paths
    import zarr
    a_dsk = np.asarray(zarr.open(a_zarr_path, mode="r")[:])
    b_dsk = np.asarray(zarr.open(b_zarr_path, mode="r")[:])

    # Apply crop bounds
    t0 = [0, 0, 0]
    if crop_bounds:
        z_s, z_e = crop_bounds["z_start"], crop_bounds["z_end"]
        y_s, y_e = crop_bounds["y_start"], crop_bounds["y_end"]
        x_s, x_e = crop_bounds["x_start"], crop_bounds["x_end"]
        tz, ty, tx = pre_reg_translation

        a_dsk = a_dsk[z_s:z_e, y_s:y_e, x_s:x_e]
        Zb, Yb, Xb = b_dsk.shape
        b_dsk = b_dsk[
            max(0, int(z_s - tz)):min(Zb, int(z_e - tz)),
            max(0, int(y_s - ty)):min(Yb, int(y_e - ty)),
            max(0, int(x_s - tx)):min(Xb, int(x_e - tx)),
        ]

    # Pad to same size
    from pyspim.util import pad_to_same_size
    if a_dsk.shape != b_dsk.shape:
        a_dsk, b_dsk = pad_to_same_size(a_dsk, b_dsk)

    # Setup bounds
    bt, br, bs = bound_translation, bound_rot_shear, bound_scale
    if transform_type == "t":
        par0 = list(t0)
        bounds = [(t - bt, t + bt) for t in t0]
    elif transform_type == "t+r":
        par0 = list(t0) + [0, 0, 0]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == "t+r+s":
        par0 = list(t0) + [0, 0, 0] + [1, 1, 1]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3 + [(1 - bs, 1 + bs)] * 3
    elif transform_type == "t+sh":
        par0 = list(t0) + [0] * 6
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6
    elif transform_type == "t+ssh":
        par0 = list(t0) + [0, 0, 0]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 3
    elif transform_type == "t+sh+s":
        par0 = list(t0) + [0] * 6 + [1, 1, 1]
        bounds = [(t - bt, t + bt) for t in t0] + [(-br, br)] * 6 + [(1 - bs, 1 + bs)] * 3
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    par0 = np.array(par0, dtype=np.float64)

    # Run optimization
    from pyspim.reg import opt
    from pyspim.util import launch_params_for_volume

    launch_par = launch_params_for_volume(a_dsk.shape, 8, 8, 8)
    a_gpu, b_gpu = cp.asarray(a_dsk), cp.asarray(b_dsk)

    if use_piecewise:
        T, res = opt.optimize_affine_piecewise(
            a_gpu, b_gpu, metric=metric, transform=transform_type,
            interp_method=interp_method, opt_method=opt_method,
            par0=par0, bounds=bounds, kernel_launch_params=launch_par, verbose=False,
        )
    else:
        T, res = opt.optimize_affine(
            a_gpu, b_gpu, metric=metric, transform=transform_type,
            interp_method=interp_method, opt_method=opt_method,
            par0=par0, bounds=bounds, kernel_launch_params=launch_par, verbose=False,
        )

    # Convert transform to list
    if hasattr(T, "get"):
        T = T.get()
    T_list = T.tolist() if hasattr(T, "tolist") else T
    cr = float(1 - res.fun)

    return {"transform_matrix": T_list, "correlation_ratio": cr}


# ---- PSF / backprojector helpers (mirrors _deconvolution.py logic) ----

def _generate_psf_pair(psf_type, fwhm_a_lat, fwhm_a_ax, fwhm_b_lat, fwhm_b_ax,
                        deskew_method, params):
    if psf_type == "gaussian":
        theta_a = _get_theta_a(deskew_method)
        theta_b = _get_theta_b(deskew_method)
        psf_a = _make_gaussian_psf(fwhm_a_lat, fwhm_a_ax, theta_a)
        psf_b = _make_gaussian_psf(fwhm_b_lat, fwhm_b_ax, theta_b)
    else:
        psf_a = _load_psf_file(params.get("psf_a_path", ""))
        psf_b = _load_psf_file(params.get("psf_b_path", ""))
    return psf_a, psf_b


def _generate_backprojector_pair(bp_type, psf_a, psf_b,
                                  bp_fwhm_a_lat, bp_fwhm_a_ax,
                                  bp_fwhm_b_lat, bp_fwhm_b_ax,
                                  deskew_method, params):
    import numpy as np
    if bp_type == "flipped_psf":
        return np.flip(psf_a).copy(), np.flip(psf_b).copy()
    elif bp_type == "gaussian":
        theta_a = _get_theta_a(deskew_method)
        theta_b = _get_theta_b(deskew_method)
        return (_make_gaussian_psf(bp_fwhm_a_lat, bp_fwhm_a_ax, theta_a),
                _make_gaussian_psf(bp_fwhm_b_lat, bp_fwhm_b_ax, theta_b))
    else:
        return (_load_psf_file(params.get("bp_a_path", "")),
                _load_psf_file(params.get("bp_b_path", "")))


def _get_theta_a(deskew_method):
    m = deskew_method.lower().replace("-", "").replace(" ", "")
    if m in ("shearwarp", "shear"):
        return 0.0
    return 45.0


def _get_theta_b(deskew_method):
    m = deskew_method.lower().replace("-", "").replace(" ", "")
    if m in ("shearwarp", "shear"):
        return 90.0
    return -45.0


def _make_gaussian_psf(fwhm_lat, fwhm_ax, theta_deg):
    import math
    import numpy as np
    from scipy import ndimage
    from napari_pyspim._psf import generate_psf_im, normalize_psf_im

    fwhm_to_sigma = 1.0 / (2 * math.sqrt(2 * math.log(2)))
    sigma_lat = fwhm_lat * fwhm_to_sigma
    sigma_ax = fwhm_ax * fwhm_to_sigma
    size = math.ceil(max(fwhm_ax, fwhm_lat) * 3)
    if size % 2 == 0:
        size += 1
    im_shape = (size, size, size)
    pars = [size / 2, size / 2, size / 2, sigma_lat, sigma_lat, sigma_ax, 1.0, 0.0]
    psf_im = generate_psf_im(pars, im_shape, "spherical")
    if theta_deg != 0.0:
        psf_im = ndimage.rotate(psf_im, theta_deg, axes=(0, 2), reshape=True,
                                 mode="constant", cval=0)
    return normalize_psf_im(psf_im)


def _load_psf_file(path):
    import numpy as np
    import tifffile
    if path.endswith(".npy"):
        return np.load(path)
    elif path.endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    try:
        return np.load(path)
    except Exception:
        return tifffile.imread(path)


def _save_result(out, save_path, output_format, shape):
    """Save deconvolution result from a zarr array."""
    import os
    import shutil
    is_mc = len(shape) > 3
    axes = "CZYX" if is_mc else "ZYX"

    if output_format == "Zarr":
        if not save_path.endswith(".zarr"):
            save_path += ".zarr"
        source_path = getattr(out.store, "root", None)
        if source_path and os.path.isdir(source_path):
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            shutil.copytree(source_path, save_path)
        else:
            import zarr
            final = zarr.open(save_path, mode="w", shape=out.shape, dtype=out.dtype,
                               chunks=out.chunks, compressor=out.compressor)
            final[:] = out[:]
        return save_path
    else:
        if not save_path.endswith(".tif") and not save_path.endswith(".tiff"):
            save_path += ".ome.tif"
        import dask.array as da
        import tifffile
        dask_data = da.from_zarr(out)
        tifffile.imwrite(save_path, dask_data, bigtiff=True, ome=True,
                         photometric="minisblack",
                         resolution=(1 / 0.1625, 1 / 0.1625),
                         metadata={
                            "axes": axes,
                            "PhysicalSizeX": 0.1625,
                            "PhysicalSizeY": 0.1625,
                            "PhysicalSizeZ": 0.1625,
                            "PhysicalSizeXUnit": 'µm',
                            "PhysicalSizeYUnit": 'µm',
                            "PhysicalSizeZUnit": 'µm',
                        },
                        tile=(1024, 1024))
        return save_path


def _save_result_array(result, save_path, output_format, shape):
    """Save deconvolution result from a numpy array."""
    import zarr
    import os
    is_mc = len(shape) > 3
    axes = "CZYX" if is_mc else "ZYX"

    if output_format == "Zarr":
        if not save_path.endswith(".zarr"):
            save_path += ".zarr"
        final = zarr.open(save_path, mode="w", shape=result.shape, dtype=result.dtype)
        final[:] = result
        return save_path
    else:
        if not save_path.endswith(".tif") and not save_path.endswith(".tiff"):
            save_path += ".ome.tif"
        import tifffile
        tifffile.imwrite(save_path, result, bigtiff=True, ome=True,
                         photometric="minisblack",
                         resolution=(1 / 0.1625, 1 / 0.1625),
                         metadata={
                            "axes": axes,
                            "PhysicalSizeX": 0.1625,
                            "PhysicalSizeY": 0.1625,
                            "PhysicalSizeZ": 0.1625,
                            "PhysicalSizeXUnit": 'µm',
                            "PhysicalSizeYUnit": 'µm',
                            "PhysicalSizeZUnit": 'µm',
                        })
        return save_path


if __name__ == "__main__":
    main()
