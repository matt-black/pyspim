from typing import Union

import cupy
import numpy

# many of the functions in this library will accept either a NumPy or CuPy
# array, this is a convenience type definition that can be used to properly
# annotate the types of these functions
NDArray = Union[numpy.ndarray, cupy.ndarray]
