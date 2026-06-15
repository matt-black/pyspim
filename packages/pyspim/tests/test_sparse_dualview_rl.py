import numpy as np
import pytest
import zarr
from pyspim.decon.sparse import deconvolve, deconvolve_chunkwise

try:
    import cupy as cp
    has_gpu = cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except ImportError:
    has_gpu = False


def _create_synthetic_data(shape=(16, 16, 16)):
    # Create a simple volume with some structure (a line in the center)
    x_true = np.zeros(shape, dtype=np.float32)
    x_true[4:12, 8, 8] = 10.0
    x_true[8, 4:12, 8] = 5.0

    # Gaussian PSFs
    psf_a = np.zeros((7, 7, 7), dtype=np.float32)
    psf_a[3, 3, :] = np.exp(-np.square(np.arange(-3, 4)) / 2.0)
    psf_a /= psf_a.sum()

    psf_b = np.zeros((7, 7, 7), dtype=np.float32)
    psf_b[:, 3, 3] = np.exp(-np.square(np.arange(-3, 4)) / 2.0)
    psf_b /= psf_b.sum()

    # Generate views by convolution
    from scipy.signal import fftconvolve
    view_a = fftconvolve(x_true, psf_a, mode="same")
    view_b = fftconvolve(x_true, psf_b, mode="same")

    # Add minor noise
    np.random.seed(42)
    view_a += np.random.normal(0, 0.01, shape).astype(np.float32)
    view_b += np.random.normal(0, 0.01, shape).astype(np.float32)

    # Keep non-negative
    view_a = np.maximum(view_a, 0.0)
    view_b = np.maximum(view_b, 0.0)

    return view_a, view_b, psf_a, psf_b, x_true


def test_sparse_rl_cpu():
    """Verify deconvolve works on CPU and regularization constraints function correctly."""
    view_a, view_b, psf_a, psf_b, _ = _create_synthetic_data()

    # Test baseline (no regularization)
    res_base = deconvolve(
        view_a=view_a,
        view_b=view_b,
        est_i=None,
        psf_a=psf_a,
        psf_b=psf_b,
        num_iter=3,
        lambda1=0.0,
        lambda2=0.0,
    )
    assert res_base.shape == view_a.shape
    assert np.all(res_base >= 0.0)

    # Test sparsity regularization (lambda1 > 0 should suppress values, making sum lower)
    res_sparse = deconvolve(
        view_a=view_a,
        view_b=view_b,
        est_i=None,
        psf_a=psf_a,
        psf_b=psf_b,
        num_iter=3,
        lambda1=1.0,
        lambda2=0.0,
    )
    assert res_sparse.shape == view_a.shape
    assert np.sum(res_sparse) < np.sum(res_base)

    # Test Hessian regularization (lambda2 > 0)
    res_hessian = deconvolve(
        view_a=view_a,
        view_b=view_b,
        est_i=None,
        psf_a=psf_a,
        psf_b=psf_b,
        num_iter=3,
        lambda1=0.0,
        lambda2=1.0,
    )
    assert res_hessian.shape == view_a.shape


@pytest.mark.skipif(not has_gpu, reason="GPU / Cupy not available")
def test_sparse_rl_gpu():
    """Verify deconvolve works on GPU with cupy arrays."""
    view_a, view_b, psf_a, psf_b, _ = _create_synthetic_data()

    view_a_gpu = cp.asarray(view_a)
    view_b_gpu = cp.asarray(view_b)
    psf_a_gpu = cp.asarray(psf_a)
    psf_b_gpu = cp.asarray(psf_b)

    res_gpu = deconvolve(
        view_a=view_a_gpu,
        view_b=view_b_gpu,
        est_i=None,
        psf_a=psf_a_gpu,
        psf_b=psf_b_gpu,
        num_iter=3,
        lambda1=0.1,
        lambda2=0.1,
    )
    assert isinstance(res_gpu, cp.ndarray)
    res_cpu = res_gpu.get()
    assert res_cpu.shape == view_a.shape
    assert np.all(res_cpu >= 0.0)


@pytest.mark.skipif(not has_gpu, reason="GPU / Cupy not available")
def test_sparse_rl_chunkwise():
    """Verify chunkwise deconvolution runs and produces output in a Zarr array."""
    view_a, view_b, psf_a, psf_b, _ = _create_synthetic_data(shape=(32, 32, 32))

    # Setup Zarr arrays (in-memory)
    store_a = zarr.storage.MemoryStore()
    store_b = zarr.storage.MemoryStore()
    store_out = zarr.storage.MemoryStore()

    z_a = zarr.array(view_a, store=store_a, chunks=(16, 16, 16))
    z_b = zarr.array(view_b, store=store_b, chunks=(16, 16, 16))
    z_out = zarr.empty((32, 32, 32), store=store_out, chunks=(16, 16, 16), dtype=np.float32)

    bp_a = psf_a[::-1, ::-1, ::-1].copy()
    bp_b = psf_b[::-1, ::-1, ::-1].copy()

    deconvolve_chunkwise(
        view_a=z_a,
        view_b=z_b,
        out=z_out,
        chunk_size=(16, 16, 16),
        overlap=(4, 4, 4),
        psf_a=psf_a,
        psf_b=psf_b,
        bp_a=bp_a,
        bp_b=bp_b,
        num_iter=2,
        epsilon=1e-5,
        lambda1=0.1,
        lambda2=0.1,
        epsilon_hess=1e-5,
        verbose=False,
    )

    out_arr = z_out[...]
    assert out_arr.shape == (32, 32, 32)
    assert np.any(out_arr > 0.0)
