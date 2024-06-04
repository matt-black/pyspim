import math

from .ortho import deskew_stage_scan as deskew_stage_scan_orthogonal
from .dispim import deskew_stage_scan as deskew_stage_scan_dispim


def deskew_stage_scan(im, pixel_size : float, step_size : float,
                      direction : int, theta : float=math.pi/4,
                      method='orthogonal', **kwargs):
    if method == 'orthogonal':
        return deskew_stage_scan_orthogonal(
            im, pixel_size, step_size, direction, theta
        )
    elif method == 'dispim':
        return deskew_stage_scan_dispim(
            im, pixel_size, step_size, direction
        )
    else:
        raise ValueError('invalid deskewing method')
