import math

import cupy

from .shear import deskew_stage_scan as deskew_stage_scan_shear
from .ortho import deskew_stage_scan as deskew_stage_scan_orthogonal
from .ortho_raw_kernel import deskew_stage_scan as deskew_stage_scan_raw
from .dispim import deskew_stage_scan as deskew_stage_scan_dispim


def deskew_stage_scan(im, pixel_size: float, step_size: float,
                      direction: int, theta: float = math.pi/4,
                      method: str = 'orthogonal',
                      **kwargs):
    gpu_in = cupy.get_array_module(im) == cupy
    if method == 'orthogonal' or method == 'ortho':
        return deskew_stage_scan_orthogonal(
            im, pixel_size, step_size, direction, theta, True
        )
    elif method == 'raw':
        return deskew_stage_scan_raw(
            im, pixel_size, step_size, direction, theta, True
        )
    elif method == 'dispim':
        return deskew_stage_scan_dispim(
            im, pixel_size, step_size, direction
        )
    elif method == 'shear':
        shr = deskew_stage_scan_shear(
            cupy.asarray(im), pixel_size, step_size, direction, 
            **kwargs
        )
        if gpu_in:
            return shr
        else:
            return shr.get()
    else:
        raise ValueError('invalid deskewing method')