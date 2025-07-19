"""Deskew stage-scanned volume into "normal" diSPIM coordinate system (see [1], esp. Figure 1).

We mimic the original Shroff lab Fiji plugin, but use a CUDA kernel for speed.
Note that interpolation is done using CUDA texture memory.

References
---
[1]  Kumar, et. al, "Using Stage- and Slit-...", doi: 10.1086/689589
"""

import os

import cupy

from .._util import create_texture_object
from ..typing import NDArray

__module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispim.cu")
with open(__module_path) as f:
    __module_txt = f.read()

__kernel_names = ("deskewTexture",)
__cuda_module = cupy.RawModule(code=__module_txt, name_expressions=__kernel_names)
__cuda_module.compile()  # throw compiler errors here, if problems


def deskew_stage_scan(
    im: NDArray, pixel_size: float, step_size: float, direction: int, **kwargs
) -> NDArray:
    """deskew_stage_scan Deskew the input volume, ``im``.

    Args:
        im (NDArray): input volume
        pixel_size (float): pixel size, in real (space) units
        step_size (float): step size, in real (space) units
        direction (int): +/-1, direction of objective movement rel. to x-axis.
        **kwargs: ``block_size`` will be passed as kernel launch parameter.

    Returns:
        NDArray
    """
    return _deskew_texture(im, pixel_size, step_size, direction, **kwargs)


def output_width(depth: int, width: int, step_size: float, pixel_size: float) -> int:
    """output_width Compute output width of deskewed volume

    Args:
        depth (int): depth of input (pre-deskewing) volume
        width (int): width of input (pre-deskewing) volume
        step_size (float): step size, in real (space) units
        pixel_size (float): pixel size, in real (space) units

    Returns:
        int
    """
    return width + round(depth * abs(step_size / pixel_size))


def _deskew_texture(
    im: NDArray,
    pixel_size: float,
    step_size: float,
    direction: int,
    block_size: int = 4,
) -> cupy.ndarray:
    assert direction == 1 or direction == -1, "direction must be +/- 1"
    depth, height, width = im.shape
    # convert image to 32-bit float
    in_type = im.dtype
    im = im.astype(cupy.float32, copy=False)
    if direction < 0:
        # in the diSPIM preprocessing plugin, they use negative
        # indexing to effectively "flip" the x-axis (see `fshifting`)
        # so to do that here, manually flip the axis and then we'll
        # reflip it back to its original orientation when we return
        im = cupy.flip(im, -1)
    # NOTE: unlike most other textures, deskewing will only "work"
    # if nearest-neighbor interpolation is used -- linear will give
    # a funny doubling pattern in the output. but also note that for
    # step sizes less than 1, this can give jagged artifacts
    tex_obj, tex_arr = create_texture_object(im, "border", "nearest", "element_type")
    depth, height, width = im.shape
    # need to rescale width s.t. entire original image
    # will fit in the deskewed output
    width = output_width(depth, width, step_size, pixel_size)
    out = cupy.zeros((depth, height, width), dtype=cupy.float32)
    # setup the kernel
    dkern = __cuda_module.get_function("deskewTexture")
    # launch kernel
    grid_x = (width + block_size - 1) // block_size
    grid_y = (height + block_size - 1) // block_size
    dkern(
        (grid_x, grid_y),
        (block_size, block_size),
        (out, tex_obj, width, height, depth, pixel_size / step_size),
    )
    del tex_obj, tex_arr
    # need to flip back to original orientation (see above)
    if direction < 0:
        out = cupy.flip(out, -1)
        im = cupy.flip(im, -1)
    # return the same data type as the input image (probably uint16)
    # NOTE: this seems to solve a CUDAIllegalMemoryAccess error that
    # can occur if the output is left as a 32-bit float
    return cupy.clip(out, 0, 2**16 - 1).astype(in_type)
