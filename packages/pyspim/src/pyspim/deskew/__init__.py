import math

import cupy
from cupyx.scipy.ndimage import zoom

from .affine import deskew_stage_scan as deskew_stage_scan_affine
from .dispim import deskew_stage_scan as deskew_stage_scan_dispim
from .ortho import deskew_stage_scan as deskew_stage_scan_orthogonal
from .orthopsf import deskew_stage_scan as deskew_stage_scan_orthopsf
from .shear import deskew_stage_scan as deskew_stage_scan_shear
from .dispim import rotate90

def deskew_stage_scan(
    im,
    pixel_size: float,
    step_size: float,
    direction: int,
    theta: float = math.pi / 4,
    method: str = "orthogonal",
    **kwargs,
):
    gpu_in = cupy.get_array_module(im) == cupy # type: ignore
    if method == "orthopsf":
        return deskew_stage_scan_orthopsf(
            im, pixel_size, step_size, direction, theta, **kwargs
        )
    elif method == "orthogeo":
        return deskew_stage_scan_orthogonal(
            im, pixel_size, step_size, direction, theta, True
        )
    elif method == "dispim":
        # NOTE: the "dispim" method does not isotropize the voxels, 
        # whereas the other 2 deskewing methods do
        # to account for this, we use the `zoom` to scale the "z" axis up
        # see `cupyx.scipy.ndimage.zoom` documentation for kwargs to pass
        dsk = deskew_stage_scan_dispim(im, pixel_size, step_size, direction)
        out = zoom(dsk, (step_size / pixel_size, 1, 1), **kwargs)
        if direction == 1:
            return out
        else:
            return rotate90(out, False, 4)
    elif method == "shear":
        shr = deskew_stage_scan_shear(
            cupy.asarray(im), pixel_size, step_size, direction, **kwargs
        )
        if gpu_in:
            return shr
        else:
            return shr.get()
    elif method == "affine":
        shr = deskew_stage_scan_affine(
            cupy.asarray(im), pixel_size, step_size, direction, **kwargs
        )
        if gpu_in:
            return shr
        else:
            return shr.get()
    else:
        raise ValueError("invalid deskewing method")
