import os
from itertools import product
from typing import Tuple

import cupy
import numpy

from .._util import launch_params_for_volume
from ..typing import NDArray

## CUDA kernel setup and module compilation

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.cu")
) as f:
    __kernel_module_txt = f.read()


def _get_kernel(
    dtype, method: str = "linear", preserve_dtype: bool = False,
) -> cupy.RawKernel:
    interp_type = method.upper()
    if dtype == cupy.uint16:
        template_U = "unsigned short"
    elif dtype == cupy.float32:
        template_U = "float"
    else:
        raise ValueError("invalid input datatype {}".format(dtype))
    if preserve_dtype == True and dtype == cupy.uint16:
        output_type = "UINT16"
        template_T = "unsigned short"
    else:
        output_type = "FLOAT32"
        template_T = "float"
    kernel_name = f"affineTransform<{template_T},{template_U}>"
    module_txt = __kernel_module_txt.format(
        interp_type=interp_type,
        output_type=output_type,
    )
    module = cupy.RawModule(
        code=module_txt,
        name_expressions=(kernel_name,),
    )
    module.compile()
    return module.get_function(kernel_name)
        

## decompositions of affine transformation matrices
def decompose_transform(A: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    xp = cupy.get_array_module(A)
    T = A[:-1, -1]
    RZS = A[:-1, :-1]
    ZS = xp.linalg.cholesky(xp.dot(RZS.T, RZS)).T
    Z = numpy.diag(ZS).copy()
    shears = ZS / Z[:, xp.newaxis]
    n = len(Z)
    S = shears[xp.triu(xp.ones((n, n)), 1).astype(bool)]
    R = xp.dot(RZS, xp.linalg.inv(ZS))
    if xp.linalg.det(R) < 0:
        Z[0] *= -1
        ZS[0] *= -1
        R = xp.dot(RZS, xp.linalg.inv(ZS))
    return T, R, Z, S


def output_shape_for_transform(
    T: NDArray, input_shape: tuple[int,int,int],
) -> Tuple[int, int, int]:
    """Calculate output shape of transformed volume.

    Args:
        T (NDArray): affine transform matrix
        input_shape (Iterable): shape of input volume (ZRC)

    Returns:
        Tuple[int,int,int]: output shape (ZRC)
    """
    t = T.get() if cupy.get_array_module(T) == cupy else T # type: ignore
    coord = list(product(*[(0, s) for s in input_shape[::-1]]))
    coord = numpy.asarray(coord).T
    coord = numpy.vstack([coord, numpy.zeros_like(coord[0, :])])
    coordT = (t @ coord)[:-1, :]
    ptp = numpy.ceil(numpy.ptp(coordT, axis=1))
    x, y, z = int(ptp[0]), int(ptp[1]), int(ptp[2])
    return z, y, x


def output_shape_for_inv_transform(
    T: NDArray, input_shape: tuple[int,int,int]
) -> Tuple[int, int, int]:
    """Calculate output shape of (inverse)-transformed volume.

    Args:
        T (NDArray): affine transform matrix (to be inverted)
        input_shape (Iterable): shape of input volume (ZRC)

    Returns:
        Tuple[int,int,int]: shape of output volume (ZRC)
    """
    xp = cupy.get_array_module(T)
    fwd = xp.linalg.inv(T).get() if xp == cupy else xp.linalg.inv(T) # type: ignore
    return output_shape_for_transform(fwd, input_shape)


def transform(
    A: NDArray,
    T: NDArray,
    interp_method: str,
    preserve_dtype: bool,
    out_shp: Tuple[int, int, int] | None,
    block_size_z: int,
    block_size_y: int,
    block_size_x: int,
    gpu_id: int = 0,
    n_gpu: int = 1,
) -> cupy.ndarray:
    """Apply affine transform to input volume.

    Args:
        A (NDArray): input volume (ZRC)
        T (NDArray): affine transform matrix (4x4), rows of matrix corresp. to XYZ
        interp_method (str): interpolation method to use when interpolating points in the transformed volume. One of ``'nearest','linear','cubspl'``.
        preserve_dtype (bool): make output datatype match that of the input (uint16), if False, output is single-precision float.
        out_shp (Tuple[int,int,int] | None): shape of output volume. if ``None``, will be calculated by this function
        block_size_z (int): size of kernel launch block, in z dimension
        block_size_y (int): size of kernel launch block, in y dimension
        block_size_x (int): size of kernel launch block, in x dimension

    Raises:
        ValueError: if input is not a ``cupy.ndarray``

    Returns:
        cupy.ndarray: transformed volume
    """
    if cupy.get_array_module(A) == cupy: # type: ignore
        kernel = _get_kernel(A.dtype, interp_method, preserve_dtype)
        if out_shp is None:
            out_shp = output_shape_for_transform(T, A.shape)
        launch_params = launch_params_for_volume(
            out_shp, block_size_z, block_size_y, block_size_x
        )
        T = cupy.asarray(T).astype(cupy.float32)
        # preallocate output and call kernel
        out_dtype = A.dtype if preserve_dtype else cupy.float32
        out = cupy.zeros(out_shp, dtype=out_dtype)
        kernel(
            launch_params[0], launch_params[1], 
            (out, A, T, *out_shp, *A.shape, gpu_id, n_gpu)
        )
        return out
    else:
        raise ValueError("only works on cupy arrays")


def _pad_amount(dim: int, chunk_dim: int) -> int:
    assert dim >= chunk_dim, f"dim : {dim:d}, chunk_dim : {chunk_dim:d}"
    n = 1
    while chunk_dim * n < dim:
        n += 1
    return chunk_dim * n - dim


def _calculate_array_chunks(
    z: int, r: int, c: int, chunk_shape: int | Tuple[int, int, int]
):
    shape = tuple([z, r, c])
    if isinstance(chunk_shape, int):
        chunk_shape = (chunk_shape, chunk_shape, chunk_shape)
    pad_size = [_pad_amount(s, cs) for s, cs in zip(shape, chunk_shape)]
    padded_shape = [s + p for s, p in zip(shape, pad_size)]
    n_chunk = [s // c for s, c in zip(padded_shape, chunk_shape)]
    chunk_mults = product(*[range(n) for n in n_chunk])
    chunk_windows = []
    for chunk_mult in chunk_mults:
        idx0 = [m * s for m, s in zip(chunk_mult, chunk_shape)]
        idx1 = [i0 + s for i0, s in zip(idx0, chunk_shape)]
        idxs = []
        for dim_idx, (i0, i1) in enumerate(zip(idx0, idx1)):
            left = i0
            right = shape[dim_idx] if i1 > shape[dim_idx] else i1
            idxs.append((left, right))
        chunk_window = [slice(l, r) for l, r in idxs]
        chunk_windows.append(chunk_window)
    return chunk_windows
