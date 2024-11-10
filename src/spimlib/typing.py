from typing import List, Tuple, Union

import cupy
import numpy

# many of the functions in this library will accept either a NumPy or CuPy
# array, this is a convenience type definition that can be used to properly
# annotate the types of these functions
NDArray = Union[numpy.ndarray, cupy.ndarray]


# type for bounding boxes
# convention here is that the tuples correspond to axes and the values
# are (lower, upper) bounds for the bounding box
# example: ((0, 10), (5, 20)) is a 2d bounding box that could be used to
#          crop `im` like im[0:10,5:20]
BBox2D = Tuple[Tuple[int,int],Tuple[int,int]]
BBox3D = Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]]

# type for specifying padding
# mostly used for deconvolution but useful for any function
# that pads an image and takes the amount of padding as input
PadType = Union[int,List[int],Tuple[int]]

# type for specifying slices for windows
SliceWindow3D = Tuple[slice,slice,slice]
SliceWindow2D = Tuple[slice,slice]

# convenience type for specifying CUDA kernel launch parameters
CuLaunchParameters = Tuple[Tuple[int,int,int], Tuple[int,int,int]]

# type for formulating optimization bounds
# can be either specified as a range, or as a margin
OptBounds = List[Tuple[float,float]]
OptBoundMargins = List[float]