"""Adaptive Median Filtering

as described in:
Weisong Zhao et al. "Sparse deconvolution improves..."
Nature Biotechnology 40, 606–617 (2022).
"""

from typing import Optional, Tuple

import cupy

from .._util import get_scipy_module
from ..typing import NDArray


def adaptive_median_filter(
    vol: NDArray, threshold, size: int | Tuple[int], footprint: Optional[NDArray] = None
) -> NDArray:
    xp = cupy.get_array_module(vol)
    sp = get_scipy_module(xp)
    median = sp.ndimage.median_filter(vol, size, footprint)
    return xp.where(vol > median * threshold, median, vol)
