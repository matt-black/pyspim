import functools
import itertools
import operator
from collections.abc import Sequence
from typing import Optional, Tuple

import cupy

from ..typing import NDArray

# useful type definitions
FRCOutput = Tuple[NDArray, NDArray]
FSCOutput = Tuple[NDArray, NDArray, NDArray]


def fourier_ring_correlation(
    im1: NDArray, im2: NDArray, bin_edges: Sequence, pixel_size: Optional[float] = None
) -> FRCOutput:
    """compute Fourier Ring Correlation for pair of input images
        pair should be checkerboard-subsampled pair of images originating
        from a single image

    :param im1: image 1
    :type im1: NDArray
    :param im2: image 2
    :type im2: NDArray
    :param bin_edges: edges of frequency bins that FRC is calculated over
    :type bin_edges: Sequence
    :param pixel_size: pixel size, in real-space units.
        if `None`, output spatial frequencies will be in units pix^{-1}
        if specified, output spatial frequencies will have correct units
    :type pixel_size: Optional[float]
    :returns: FRC values and corresponding spatial frequencies
    :rtype: Tuple[NDArray,NDArray]
    """
    pixel_size = 1.0 if pixel_size is None else pixel_size
    assert len(im1.shape) == 2 and len(im2.shape) == 2, "both inputs must be 2D images"
    # figure out auto-dispatch
    xp = cupy.get_array_module(im1)
    assert xp == cupy.get_array_module(im2), (
        "both images (`im1` & `im2`) must be on same device"
    )
    # compute fourier transform
    fft1 = xp.fft.fftshift(xp.fft.fft2(im1))
    fft2 = xp.fft.fftshift(xp.fft.fft2(im2))
    # figure out center coordinate of fft
    ysze, xsze = fft1.shape
    yc, xc = ysze / 2.0, xsze / 2.0
    # formulate a grid of coordinates & use to calculate radial
    # distance from center at each point in the image
    grid = xp.meshgrid(xp.arange(xsze) - xc, xp.arange(ysze) - yc, indexing="xy")
    R = xp.sqrt(functools.reduce(operator.add, map(xp.square, grid)))
    # calculate spatial frequencies (in real units) being probed
    nyq_frq = xp.floor(im1.shape[0] * pixel_size / 2.0)  # Nyquist frequency
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    spt_frq = bin_centers / nyq_frq
    frc, npts = __fourier_correlation(fft1, fft2, R, bin_edges)
    return frc, spt_frq


def fourier_shell_correlation(
    vol1: NDArray,
    vol2: NDArray,
    bin_edges: Sequence,
    voxel_size: Optional[float] = None,
) -> FSCOutput:
    """compute Fourier Shell Correlation for pair of input volumes
        pair should be checkerboard-subsampled pair of volumes originating
        from a single volume

    :param vol1: volume 1
    :type vol1: NDArray
    :param vol2: volume 2
    :type vol2: NDArray
    :param bin_edges:
    :type bin_edges: Sequence
    :param voxel_size:
    :type voxel_size: Optional[float]
    :returns:
    :rtype: Tuple[NDArray,NDArray,NDArray]
    """
    voxel_size = 1.0 if voxel_size is None else voxel_size
    assert len(vol1.shape) == 3 and len(vol2.shape) == 3, (
        "both inputs must be 3D volumes"
    )
    # figure out auto-dispatch
    xp = cupy.get_array_module(vol1)
    assert xp == cupy.get_array_module(vol2), (
        "both images (`vol1` & `vol2`) must be on same device"
    )
    # compute fourier transform
    fft1 = xp.fft.fftshift(xp.fft.fftn(vol1))
    fft2 = xp.fft.fftshift(xp.fft.fftn(vol2))
    # figure out center coordinate of fft
    zsze, ysze, xsze = fft1.shape
    zc, yc, xc = zsze / 2.0, ysze / 2.0, xsze / 2.0
    # formulate a grid of coordinates & use to calculate radial
    # distance from center at each voxel in volume
    grid = xp.meshgrid(
        xp.arange(xsze) - xc, xp.arange(ysze) - yc, xp.arange(zsze) - zc, indexing="xy"
    )
    R = xp.sqrt(functools.reduce(operator.add, map(xp.square, grid)))
    # calculate spatial frequencies (in real units) being probed
    nyq_frq = xp.floor(vol1.shape[0] * voxel_size / 2.0)  # Nyquist frequency
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
    spt_frq = bin_centers / nyq_frq
    fsc, npts = __fourier_correlation(fft1, fft2, R, bin_edges)
    return fsc, spt_frq, npts


def __fourier_correlation(
    fft1: NDArray, fft2: NDArray, R: NDArray, bin_edges: Sequence
) -> Tuple[NDArray, NDArray]:
    """do the actual fourier ring/shell calculation"""
    xp = cupy.get_array_module(fft1)
    # digitize the image s.t. pixel values of the image
    # are the corresp. histogram bin
    B = xp.digitize(R, bin_edges, right=False)
    f1_dot_f2 = xp.zeros(len(bin_edges) - 1, dtype=float)
    f1sq = xp.zeros_like(f1_dot_f2)
    f2sq = xp.zeros_like(f1_dot_f2)
    npts = xp.zeros_like(f1_dot_f2)
    # iterate over bins, calculating correlation @ each
    for b in range(1, len(bin_edges)):
        # select out relevant parts of FFTs
        f1, f2 = fft1[B == b], fft2[B == b]
        # numerator is F_1(r) \dot F_2(r)*
        f1_dot_f2[b - 1] = xp.sum(f1 * xp.conjugate(f2)).real
        # denominator terms
        f1sq[b - 1] = xp.sum(xp.abs(f1) ** 2)
        f2sq[b - 1] = xp.sum(xp.abs(f2) ** 2)
        # save # of pts in each bin
        # NOTE: by defn, len(f1) == len(f2), so just save 1
        npts[b - 1] = len(f1)
    # the actual calculation
    fc = xp.abs(f1_dot_f2) / xp.sqrt(f1sq * f2sq)
    fc[fc == xp.inf] = 0.0
    fc = xp.nan_to_num(fc)
    return fc, npts


def single_section_fsc(
    vol1: NDArray,
    vol2: NDArray,
    bin_edges: Sequence,
    alpha_lo,
    alpha_hi,
    voxel_size: Optional[float] = None,
):
    voxel_size = 1.0 if voxel_size is None else voxel_size
    assert len(vol1.shape) == 3 and len(vol2.shape) == 3, (
        "both inputs must be 3D volumes"
    )
    xp = cupy.get_array_module(vol1)
    assert xp == cupy.get_array_module(vol2), (
        "both images (`vol1` & `vol2`) must be on same device"
    )
    # calculate FFTs
    fft1 = xp.fft.fftshift(xp.fft.fftn(vol1))
    fft2 = xp.fft.fftshift(xp.fft.fftn(vol2))
    # figure out center coordinates
    zsze, ysze, xsze = fft1.shape
    xc, yc, zc = xsze / 2.0, ysze / 2.0, zsze / 2.0
    raise NotImplementedError("todo")


def sectioned_fourier_shell_correlation(
    vol1: NDArray,
    vol2: NDArray,
    r_bin_edges: Sequence,
    alpha_bin_edges: Sequence,
    voxel_size: float,
):
    voxel_size = 1.0 if voxel_size is None else voxel_size
    assert len(vol1.shape) == 3 and len(vol2.shape) == 3, (
        "both inputs must be 3D volumes"
    )
    # figure out auto-dispatch
    xp = cupy.get_array_module(vol1)
    assert xp == cupy.get_array_module(vol2), (
        "both images (`vol1` & `vol2`) must be on same device"
    )
    # compute fourier transform
    fft1 = xp.fft.fftshift(xp.fft.fftn(vol1))
    fft2 = xp.fft.fftshift(xp.fft.fftn(vol2))
    # figure out center coordinate of fft
    zsze, ysze, xsze = fft1.shape
    zc, yc, xc = zsze / 2.0, ysze / 2.0, xsze / 2.0
    # formulate a grid of coordinates & use to calculate radial
    # distance from center at each voxel in volume
    grid = xp.meshgrid(
        xp.arange(xsze) - xc, xp.arange(ysze) - yc, xp.arange(zsze) - zc, indexing="xy"
    )
    R = xp.sqrt(functools.reduce(operator.add, map(xp.square, grid)))
    A = xp.arccos(grid[-1] / R)
    nyq_frq = xp.floor(vol1.shape[0] * voxel_size / 2.0)  # Nyquist frequency
    r_bin_centers = (r_bin_edges[1:] + r_bin_edges[:-1]) / 2.0
    spt_frq = r_bin_centers / nyq_frq
    # hist_vals = xp.meshgrid(spt_frq, a_bin_centers)
    fsc, npts = __sectioned_fourier_correlation(
        fft1, fft2, R, A, r_bin_edges, alpha_bin_edges
    )
    return fsc, spt_frq, npts


def __sectioned_fourier_correlation(fft1, fft2, R, A, r_bin_edges, a_bin_edges):
    xp = cupy.get_array_module(fft1)
    Br = xp.digitize(R, r_bin_edges, right=False)
    Ba = xp.digitize(A, a_bin_edges, right=False)
    f1_dot_f2 = xp.zeros((len(r_bin_edges) - 1, len(a_bin_edges) - 1), dtype=float)
    f1sq = xp.zeros_like(f1_dot_f2)
    f2sq = xp.zeros_like(f1_dot_f2)
    npts = xp.zeros_like(f1_dot_f2)
    bin_iterator = itertools.product(
        range(1, len(r_bin_edges)), range(1, len(a_bin_edges))
    )
    for br, ba in bin_iterator:
        f1 = fft1[xp.logical_and(Br == br, Ba == ba)]
        f2 = fft2[xp.logical_and(Br == br, Ba == ba)]
        f1_dot_f2[br - 1, ba - 1] = xp.sum(f1 * xp.conjugate(f2)).real
        # denominator
        f1sq[br - 1, ba - 1] = xp.sum(xp.abs(f1) ** 2)
        f2sq[br - 1, ba - 1] = xp.sum(xp.abs(f2) ** 2)
        npts[br - 1, ba - 1] = len(f1)
    fc = xp.abs(f1_dot_f2) / xp.sqrt(f1sq * f2sq)
    fc[fc == xp.inf] = 0.0
    fc = xp.nan_to_num(fc)
    return fc, npts
