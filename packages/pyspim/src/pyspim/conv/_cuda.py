import os
from typing import Tuple

import cupy

from ..typing import CuLaunchParameters

## load convolution as cupy.RawModule
__conv3_module_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "conv3.cu"
)
with open(__conv3_module_path) as f:
    __conv_module_txt = f.read()


__kernel_names = (
    "convZKernel<unsigned short>",
    "convZKernel<float>",
    "convRKernel<unsigned short>",
    "convRKernel<float>",
    "convCKernel<unsigned short>",
    "convCKernel<float>",
)


def _cuda_parameters_for_kernel_radius(kernel_radius: int):
    assert kernel_radius in (3, 7, 15, 31, 63, 127), (
        "kernel radius must be one of (3, 7, 15, 31, 63, 127)"
    )
    if kernel_radius == 3:
        z_blockdim_x, z_blockdim_y = 4, 32
        z_result_steps, z_border_pix = 8, 1
        r_blockdim_x, r_blockdim_y = 32, 4
        r_result_steps, r_border_pix = 8, 1
        c_blockdim_x, c_blockdim_z = 32, 4
        c_result_steps, c_border_pix = 8, 1
    elif kernel_radius == 7:
        z_blockdim_x, z_blockdim_y = 8, 32
        z_result_steps, z_border_pix = 8, 1
        r_blockdim_x, r_blockdim_y = 32, 8
        r_result_steps, r_border_pix = 8, 1
        c_blockdim_x, c_blockdim_z = 32, 8
        c_result_steps, c_border_pix = 8, 1
    elif kernel_radius == 15:
        z_blockdim_x, z_blockdim_y = 16, 16
        z_result_steps, z_border_pix = 8, 1
        r_blockdim_x, r_blockdim_y = 16, 16
        r_result_steps, r_border_pix = 8, 1
        c_blockdim_x, c_blockdim_z = 16, 16
        c_result_steps, c_border_pix = 8, 1
    elif kernel_radius == 31:
        z_blockdim_x, z_blockdim_y = 32, 8
        z_result_steps, z_border_pix = 8, 1
        r_blockdim_x, r_blockdim_y = 8, 32
        r_result_steps, r_border_pix = 8, 1
        c_blockdim_x, c_blockdim_z = 8, 32
        c_result_steps, c_border_pix = 8, 1
    elif kernel_radius == 63:
        z_blockdim_x, z_blockdim_y = 64, 2
        z_result_steps, z_border_pix = 8, 1
        r_blockdim_x, r_blockdim_y = 2, 64
        r_result_steps, r_border_pix = 8, 1
        c_blockdim_x, c_blockdim_z = 2, 64
        c_result_steps, c_border_pix = 8, 1
    elif kernel_radius == 127:
        z_blockdim_x, z_blockdim_y = 128, 1
        z_result_steps, z_border_pix = 4, 1
        r_blockdim_x, r_blockdim_y = 1, 128
        r_result_steps, r_border_pix = 4, 1
        c_blockdim_x, c_blockdim_z = 1, 128
        c_result_steps, c_border_pix = 4, 1
    else:
        raise ValueError("invalid kernel size")
    return (
        (z_blockdim_x, z_blockdim_y, z_result_steps, z_border_pix),
        (r_blockdim_x, r_blockdim_y, r_result_steps, r_border_pix),
        (c_blockdim_x, c_blockdim_z, c_result_steps, c_border_pix),
    )


def make_conv_module(
    kernel_radius: int,
    module_txt: str = __conv_module_txt,
    name_expr: Tuple[str] = __kernel_names,
) -> cupy.RawModule:
    """make_conv_module Generate ``cupy.RawModule`` for separable convolution

    Args:
        kernel_radius (int): radius of convolution kernel
        module_txt (str, optional): C++ code to be compiled. Defaults to __conv_module_txt.
        name_expr (Tuple[str], optional): named expressions that are exposed by the module. Defaults to __kernel_names.

    Returns:
        cupy.RawModule
    """
    z_pars, r_pars, c_pars = _cuda_parameters_for_kernel_radius(kernel_radius)
    # we only unroll loops on relatively small kernels
    if kernel_radius < 127:
        pragma_unroll = "#pragma unroll"
    else:
        pragma_unroll = ""
    z_blockdim_x, z_blockdim_y, z_result_steps, z_border_pix = z_pars
    r_blockdim_x, r_blockdim_y, r_result_steps, r_border_pix = r_pars
    c_blockdim_x, c_blockdim_z, c_result_steps, c_border_pix = c_pars
    txt = module_txt.format(
        kernel_radius=kernel_radius,
        pragma_unroll=pragma_unroll,
        z_blockdim_x=z_blockdim_x,
        z_blockdim_y=z_blockdim_y,
        z_result_steps=z_result_steps,
        z_border_pix=z_border_pix,
        r_blockdim_x=r_blockdim_x,
        r_blockdim_y=r_blockdim_y,
        r_result_steps=r_result_steps,
        r_border_pix=r_border_pix,
        c_blockdim_x=c_blockdim_x,
        c_blockdim_z=c_blockdim_z,
        c_result_steps=c_result_steps,
        c_border_pix=c_border_pix,
    )
    module = cupy.RawModule(code=txt, name_expressions=name_expr)
    module.compile()
    return module


def _launch_pars_z(
    sze_z: int,
    sze_r: int,
    sze_c: int,
    z_result_steps: int,
    z_blockdim_x: int,
    z_blockdim_y: int,
):
    zrsTzbx = z_result_steps * z_blockdim_x
    gx = sze_z // zrsTzbx + min(1, sze_z % zrsTzbx)
    gy = sze_r // z_blockdim_y + min(1, sze_r % z_blockdim_y)
    gz = sze_c
    return (gx, gy, gz), (z_blockdim_x, z_blockdim_y, 1)


def _launch_pars_r(
    sze_z: int,
    sze_r: int,
    sze_c: int,
    r_result_steps: int,
    r_blockdim_x: int,
    r_blockdim_y: int,
):
    rrsTrby = r_result_steps * r_blockdim_y
    gx = sze_z // r_blockdim_x + min(1, sze_z % r_blockdim_x)
    gy = sze_r // rrsTrby + min(1, sze_r % rrsTrby)
    gz = sze_c
    return (gx, gy, gz), (r_blockdim_x, r_blockdim_y, 1)


def _launch_pars_c(
    sze_z: int,
    sze_r: int,
    sze_c: int,
    c_result_steps: int,
    c_blockdim_x: int,
    c_blockdim_z: int,
):
    crsTcbz = c_result_steps * c_blockdim_z
    gx = sze_z // c_blockdim_x + min(1, sze_z % c_blockdim_x)
    gy = sze_r
    gz = sze_c // crsTcbz + min(1, sze_r % crsTcbz)
    return (gx, gy, gz), (c_blockdim_x, 1, c_blockdim_z)


def calc_launch_params(
    kernel_radius: int, vol_shape: Tuple[int, int, int]
) -> CuLaunchParameters:
    """calc_launch_params Launch parameters for convolution kernel.

    Args:
        kernel_radius (int): radius of convolution kernel
        vol_shape (Tuple[int,int,int]): dimensions of volume to be convolved over (ZRC)

    Returns:
        CuLaunchParameters
    """
    z_pars, r_pars, c_pars = _cuda_parameters_for_kernel_radius(kernel_radius)
    sz, sr, sc = vol_shape
    z_blockdim_x, z_blockdim_y, z_result_steps, _ = z_pars
    launch_z = _launch_pars_z(sz, sr, sc, z_result_steps, z_blockdim_x, z_blockdim_y)
    r_blockdim_x, r_blockdim_y, r_result_steps, _ = r_pars
    launch_r = _launch_pars_r(sz, sr, sc, r_result_steps, r_blockdim_x, r_blockdim_y)
    c_blockdim_x, c_blockdim_z, c_result_steps, _ = c_pars
    launch_c = _launch_pars_c(sz, sr, sc, c_result_steps, c_blockdim_x, c_blockdim_z)
    return launch_z, launch_r, launch_c


_TupleCLP = Tuple[CuLaunchParameters, CuLaunchParameters, CuLaunchParameters]


def convolve_3d(
    x: cupy.ndarray,
    kernel_z: cupy.ndarray,
    kernel_y: cupy.ndarray,
    kernel_x: cupy.ndarray,
    cuda_module: cupy.RawModule | None = None,
    launch_pars: _TupleCLP | None = None,
) -> cupy.ndarray:
    """convolve_3d Separable 3D convolution of input volume with z, y, x kernels.

    Args:
        x (cupy.ndarray): input volume
        kernel_z (cupy.ndarray): convolution kernel in z-direction
        kernel_y (cupy.ndarray): convolution kernel in y-direction
        kernel_x (cupy.ndarray): convolution kernel in x-direction
        cuda_module (cupy.RawModule | None, optional): ``cupy.RawModule`` that implements the convolution. Defaults to None.
        launch_pars (_TupleCLP | None, optional): kernel launch parameters. Defaults to None.

    Returns:
        cupy.ndarray
    """
    x = cupy.asarray(x, order="F")  # kernel only works w. column-major
    # make sure all kernels are the same size
    krx = kernel_x.shape[0] // 2
    kry = kernel_y.shape[0] // 2
    krz = kernel_z.shape[0] // 2
    assert krx == kry and kry == krz, "all kernels must be same size"
    # if the user doesn't pass in a (pre-compiled) cuda module, we have
    # to do the compilation, here
    # NOTE: modules are compiled based on the input kernel radius & it is up
    # to the user to make sure that the module they pass in was compiled for
    # the appropriate kernel size, if they do so
    if cuda_module is None:
        cuda_module = make_conv_module(krx)
    # convert input to float32 and pre-allocate output array
    out = cupy.zeros_like(x, dtype=cupy.float32)
    # determine launch parameters for this volume
    shp = tuple(list(x.shape))
    if launch_pars is None:
        launch_z, launch_r, launch_c = calc_launch_params(krx, shp)
    else:
        launch_z, launch_r, launch_c = launch_pars
    # do the convolution (Z -> R -> C)
    dtype_str = "unsigned short" if x.dtype == cupy.uint16 else "float"
    convz = cuda_module.get_function(f"convZKernel<{dtype_str:s}>")
    convz(launch_z[0], launch_z[1], (out, x, kernel_z, *shp))
    convr = cuda_module.get_function(f"convRKernel<{dtype_str:s}>")
    convr(launch_r[0], launch_r[1], (out, x, kernel_y, *shp))
    convc = cuda_module.get_function(f"convCKernel<{dtype_str:s}>")
    convc(launch_c[0], launch_c[1], (out, x, kernel_x, *shp))
    return out
