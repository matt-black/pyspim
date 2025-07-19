"""Single-view Richardson Lucy deconvolution using FFT-space convolution.

References
---
[1] Bertero & Boccacci, "A simple method...", doi:10.1051/0004-6361:20052717
"""

from typing import Optional

import cupy
from cuypx.scipy.signal import fftconvolve as fftconv_gpu
from scipy.signal import fftconvolve as fftconv_cpu
from tqdm.auto import trange

from ..._util import supported_float_type
from ...typing import NDArray, PadType


def richardson_lucy(
    image: NDArray,
    psf: NDArray,
    bp: NDArray,
    num_iter: int = 50,
    boundary_correction: bool = True,
    epsilon: Optional[float] = None,
    boundary_padding: Optional[int] = None,
    boundary_sigma: float = 1e-2,
    verbose: bool = False,
) -> NDArray:
    """richardson_lucy Richardson-Lucy deconvolution.

    Args:
        image (NDArray): input volume to be deconvolved
        psf (NDArray): point spread function
        bp (NDArray): back projector for deconvolution
        num_iter (int, optional): number of iterations. Defaults to 50.
        boundary_correction (bool, optional): whether or not to do boundary correction. Defaults to True.
        epsilon (Optional[float], optional): small parameter to prevent division by zero. Defaults to None.
        boundary_padding (Optional[int], optional): zero-padding for boundary correction. Defaults to None.
        boundary_sigma (float, optional): significance level for pixels when doing boundary correction. Defaults to 1e-2.
        verbose (bool, optional): show a progress bar. Defaults to False.

    Raises:
        NotImplementedError: uncorrected version of RL deconvolution.

    Returns:
        NDArray
    """
    if boundary_correction:
        return richardson_lucy_boundcorr(
            image,
            psf,
            bp,
            num_iter,
            epsilon,
            boundary_padding,
            boundary_sigma,
            init_constant=False,
            verbose=verbose,
        )
    else:
        raise NotImplementedError("TODO")


def richardson_lucy_boundcorr(
    image: NDArray,
    psf: NDArray,
    bp: Optional[NDArray] = None,
    num_iter: int = 50,
    epsilon: float = 1e-4,
    zero_padding: Optional[PadType] = None,
    boundary_sigma: float = 1e-2,
    init_constant: bool = False,
    verbose: bool = False,
) -> NDArray:
    """richardson_lucy_boundcorr Richardson-Lucy deconvolution with boundary correction described in [1], this method reduces Gibbs oscillations that occur due to boundary effects when doing RL deconvolution.

    Args:
        image (NDArray): input volume to be deconvolved
        psf (NDArray): point spread function
        bp (Optional[NDArray], optional): backprojector. Defaults to None, if so will used mirrored PSF.
        num_iter (int, optional): number of iterations. Defaults to 50.
        epsilon (float, optional): small parameter to prevent division by zero. Defaults to 1e-4.
        zero_padding (Optional[PadType], optional): amount of zero-padding. Defaults to None.
        boundary_sigma (float, optional): significance level for pixels when doing boundary correction. Defaults to 1e-2.
        init_constant (bool, optional): whether to make the inital guess a constant array, (``True``) or (A+B)/2 (``False``). Defaults to False.
        verbose (bool, optional): show progress bar. Defaults to False.

    Returns:
        NDArray
    """
    assert len(image.shape) == 2 or len(image.shape) == 3, (
        "RL deconvolution only implemented for 2- and 3d inputs"
    )
    xp = cupy.get_array_module(image)
    # make sure all inputs are floats
    float_type = supported_float_type(image.dtype)
    image = image.astype(float_type, copy=False)
    psf = psf.astype(float_type, copy=False)
    # default back-projector is mirrored PSF
    if bp is None:
        bp = xp.ascontiguousarray(psf[::-1, ::-1, ::-1])
    # figure out padding
    if zero_padding is None:  # use default value from paper (2N)
        # default is to pad such that an NxN image is 2Nx2N
        pad = tuple([(d // 2, d // 2) for d in image.shape])
    else:
        if isinstance(zero_padding, int):
            pad = tuple([(zero_padding, zero_padding) for _ in image.shape])
        else:
            assert len(zero_padding) == len(image.shape), (
                "zero padding must be specified for all dimensions of input"
            )
            pad = tuple([(p, p) for p in zero_padding])
    # define utility function for doing convolution
    # NOTE: by default `fftconvolve` wants the larger array as first arg
    # but for clarity, I like specifying the PSF first
    if xp == cupy:
        conv = lambda k, i: fftconv_gpu(i, k, mode="same")
    else:
        conv = lambda k, i: fftconv_cpu(i, k, mode="same")
    # generate the mask that we'll use to determine $\overbar{alpha}$
    mask = xp.ones_like(image)
    # pad the image and the mask with zeros
    image = xp.pad(image, padding=pad, mode="constant", constant_values=0)
    mask = xp.pad(mask, padding=pad, mode="constant", constant_values=0)
    # calculate the window, $\overbar{w}(n')
    alpha = conv(bp, mask)
    window = xp.where(alpha > boundary_sigma, 1.0 / alpha, 0.0)
    # initialization conditions
    # if *not* init_constant, just start with the data
    if init_constant:
        est = xp.ones_like(image) * xp.mean(image)
    else:
        est = image
    # Lucy-Richardson iterations
    for _ in trange(num_iter) if verbose else range(num_iter):
        con = conv(psf, est)
        est = xp.multiply(
            xp.multiply(window, est),  # conv(window, est)
            conv(bp, xp.where(con < epsilon, 0, image / con)),
        )
    # trim out padding
    if len(est) == 2:
        return est[pad[0][0] : -pad[0][1], pad[1][0] : -pad[1][1]]
    else:
        return est[
            pad[0][0] : -pad[0][1], pad[1][0] : -pad[1][1], pad[2][0] : -pad[2][1]
        ]
