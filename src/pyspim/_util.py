"""private utilities
"""
import json
import types
import typing
from collections.abc import Iterable
from functools import partial

import cupy
import numpy

# scipy
import scipy
import cupyx.scipy
import cupyx.scipy.fft as fftgpu
import scipy.fft as fftcpu
from scipy import ndimage as ndi_cpu
from cupyx.scipy import ndimage as ndi_gpu
# cupy.cuda
from cupy.cuda import texture
from cupy.cuda import runtime

from .typing import NDArray, BBox2D, BBox3D, CuLaunchParameters


def get_fft_module(arr_or_xp) -> types.ModuleType:
    """get_fft_module FFT Module for input arguments

    Args:
        arr_or_xp (NDArray or numpy/cupy module): input to determine whether scipy (cpu) or cupyx (gpu) should be used.

    Returns:
        types.ModuleType: `scipy.fft` or `cupyx.scipy.fft`
    """
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return fftcpu
    else:
        return fftgpu


def get_scipy_module(arr_or_xp) -> types.ModuleType:
    """get_scipy_module Scipy module for input arguments.

    Args:
        arr_or_xp (NDArray or numpy/cupy module): input to determine whether scipy (cpu) or cupyx (gpu) should be used.

    Returns:
        types.ModuleType: `scipy` or `cupyx.scipy`
    """
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return scipy
    else:
        return cupyx.scipy


def get_ndimage_module(arr_or_xp) -> types.ModuleType:
    """get_ndimage_module NDImage module for input arguments

    Args:
        arr_or_xp (NDArray or numpy/cupy module): input to determine whether scipy (cpu) or cupyx (gpu) should be used.

    Returns:
        types.ModuleType: `scipy.ndimage` or `cupyx.scipy.ndimage`
    """
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return ndi_cpu
    else:
        return ndi_gpu


## misc. support functions
def supported_float_type(input_dtype, allow_complex : bool=False):
    """supported_float_type Return an appropriate floating-point dtype for given dtype.
        float32, float64, complex64, complex128 are preserved.
        float16 is promoted to float32.
        complex256 is demoted to complex128.
        Other types are cast to float64.
    
    Args:
        input_dtype (_type_): the input dtype.
        allow_complex (bool, optional): let this function return a complex dtype. Defaults to False.

    Raises:
        ValueError: `allow_complex=False` and input is complex

    Returns:
        dtype
    """
    new_float_type = {
        # preserved types
        'f': cupy.float32,     # float32
        'd': cupy.float64,     # float64
        'F': cupy.complex64,   # complex64
        'D': cupy.complex128,  # complex128
        # promoted float types
        'e': cupy.float32,     # float16
        # truncated float types
        'g': cupy.float64,     # float128 (doesn't exist on windows)
        'G': cupy.complex128,  # complex256 (doesn't exist on windows)
        # integer types that can be exactly represented in float32
        'b': cupy.float32,     # int8
        'B': cupy.float32,     # uint8
        'h': cupy.float32,     # int16
        'H': cupy.float32,     # uint16
        '?': cupy.float32,     # bool
    }
    if isinstance(input_dtype, Iterable) and not isinstance(input_dtype, str):
        return cupy.result_type(*(supported_float_type(d) for d in input_dtype))
    input_dtype = cupy.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, cupy.float64)


def is_floating_point(x : NDArray) -> bool:
    """is_floating_point Boolean indicator for if input is a floating point dtype

    Args:
        x (NDArray): array to check type of

    Returns:
        bool
    """
    xp = cupy.get_array_module(x)
    return x.dtype in (xp.float16, xp.float32, xp.float64)


## padding/shape utilities
def shape_for_divisible(shp : Iterable[int], *div) -> typing.List[int]:
    """shape_for_divisible Compute the shape >= input that is divisible in each dimension.

    Args:
        shp (Iterable[int]): shape of input volume
        *div (int): divisor in each dimension
    Returns:
        typing.List[int]: largest size smaller than dimension divisible by input arg
    """
    out = []
    assert len(shp) == len(div), \
        "must give one division/dimension"
    for dim, d in zip(shp, div):
        if dim % d == 0:
            out.append(dim)
        else:
            out.append(dim + d - dim % d)
    return out


def pad_to_shape_right(vol : NDArray, shape : Iterable[int], **kwargs) -> NDArray:
    """pad_to_shape_right Pad input volume only on RHS so that it has specified `shape`.

    Args:
        vol (NDArray): input volume
        shape (Iterable[int]): desired output shape
        **kwargs: keyword arguments passed to `numpy.pad` or `cupy.pad`
    Returns:
        NDArray: 
    """
    xp = cupy.get_array_module(vol)
    s_, v_ = list(shape), []
    for i, _ in enumerate(s_):
        v_.append(vol.shape[i])
    pads = [s-v for s, v in zip(s_, v_)]
    assert all([p>=0 for p in pads]), "all dims must be >= output shape"
    pads = tuple([tuple([0, p]) for p in pads])
    return xp.pad(vol, pads, **kwargs)


def unpad_to_shape_right(vol : NDArray, shape : Iterable[int]) -> NDArray:
    s_, v_ = list(shape), []
    for i, _ in enumerate(s_):
        v_.append(vol.shape[i])
    pads = [v-s for v, s in zip(v_, s_)]
    assert all([p>=0 for p in pads]), \
        "output shape must be lte than current in all dimensions"
    starts = [0 for _ in pads]
    ends = [v-p for v, p in zip(v_, pads)]
    slices = tuple([slice(s, e) for s, e in zip(starts, ends)])
    return vol[slices]


def pad_to_same_size(a : NDArray, b : NDArray, style='center', **kwargs) -> \
    typing.Tuple[NDArray, NDArray]:
    if style == 'center':
        pad_fun = pad_to_shape_center
    elif style == 'right':
        pad_fun = pad_to_shape_right
    else:
        raise ValueError('invalid padding style')
    if len(a.shape) == 3:
        za, ya, xa = a.shape
        zb, yb, xb = b.shape
        z, y, x = max([za, zb]), max([ya, yb]), max([xa, xb])
        fn = partial(pad_fun, shape=[z, y, x], **kwargs)
        return fn(a), fn(b)
    else:
        assert len(a.shape) == 2, \
            'must be 2d image if not a 3d volume, is {:d}d'.format(len(a.shape))
        ya, xa = a.shape
        yb, xb = b.shape
        y, x = max([ya, yb]), max([xa, xb])
        fn = partial(pad_fun, shape=[y,x], **kwargs)
        return fn(a), fn(b) 


def pad_to_shape_center(vol : NDArray, shape, **kwargs) -> NDArray:
    """pad input volume to specified shape
        padding is done such that original volume is centered in the output
        `**kwargs` are passed to underlying `numpy.pad` or `cupy.pad` function

    :param vol: input volume
    :type vol: NDArray
    :param shape: desired output shape
    :type shape: tuple or Iterable
    :returns: padded volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(vol)
    s_, v_ = list(shape), []
    for i, s in enumerate(s_):
        v_.append(vol.shape[i])
    pads = [s-v for s, v in zip(s_, v_)]
    assert all([p>=0 for p in pads]), "all dims must be >= output shape"
    left_pads = [p//2 for p in pads]
    right_pads = [p//2 + p%2 for p in pads]
    pads = tuple([tuple([l, r]) for l, r in zip(left_pads, right_pads)])
    return xp.pad(vol, pads, **kwargs)


def unpad_to_shape_center(vol : NDArray, shape) -> NDArray:
    """inverse of `pad_to_shape`, gets rid of padding at volume edges

    :param vol: input, padded volume
    :type vol: NDArray
    :param shape: desired output shape
    :type shape: tuple or Iterable
    :returns: unpadded volume
    :rtype: NDArray
    """
    s_, v_ = list(shape), []
    for i, _ in enumerate(s_):
        v_.append(vol.shape[i])
    pads = [v-s for v, s in zip(v_, s_)]
    assert all([p>=0 for p in pads]), \
        "output shape must be lte than current in all dimensions"
    starts = [p//2 for p in pads]
    ends = [v-(p//2 + p%2) for v, p in zip(v_, pads)]
    slices = tuple([slice(s, e) for s, e in zip(starts, ends)])
    return vol[slices]


def pad_to_same_size_center(a : NDArray, b : NDArray,
                            **kwargs) -> typing.Tuple[NDArray,NDArray]:
    """pad input volumes so that they are the same size
        finds smallest possible padding in each dimension to add so that the
        shared size is as small as possible
        `**kwargs` are passed to underlying pad function
        (`numpy.pad` or `cupy.pad` depending on input)

    :param a: array 1
    :type a: NDArray
    :param b: array 2
    :type b: NDArray
    :returns: padded versions of input volumes with common shape
    :rtype: tuple[NDArray]
    """
    if len(a.shape) == 3:
        za, ya, xa = a.shape
        zb, yb, xb = b.shape
        z, y, x = max([za, zb]), max([ya, yb]), max([xa, xb])
        fn = partial(pad_to_shape_center, shape=[z, y, x], **kwargs)
        return fn(a), fn(b)
    else:
        assert len(a.shape) == 2, 'must be 2d image if not a 3d volume'
        ya, xa = a.shape
        yb, xb = b.shape
        y, x = max([ya, yb]), max([xa, xb])
        fn = partial(pad_to_shape_center, shape=[y,x], **kwargs)
        return fn(a), fn(b)


def center_crop(vol : NDArray, *args) -> NDArray:
    """crop out the center of the input volume

    :param vol: input volume
    :type vol: NDArray
    :returns: crop of input volume
    :rtype: NDArray
    """
    # unpack args into crop_z, crop_r, crop_c
    n_dim = len(vol.shape)
    assert n_dim == 2 or n_dim == 3, \
        "input must be 2D image or 3D volume"
    if len(args) == 1:
        crop_z, crop_r, crop_c = args[0], args[0], args[0]
    else:
        assert len(args) == n_dim, \
            "must specify either 1 or `n_dim` crop inputs"
        if n_dim == 2:
            crop_r, crop_c = args[0], args[1]
        else:  # n_dim == 3
            crop_z, crop_r, crop_c = args[0], args[1], args[2]
    if n_dim == 2:
        size_r, size_c = vol.shape
    else:
        size_z, size_r, size_c = vol.shape
        crop_z = crop_z if crop_z < size_z else size_z
        sz = size_z // 2 - crop_z // 2
    crop_r = crop_r if crop_r < size_r else size_r
    crop_c = crop_c if crop_c < size_c else size_c
    sr = size_r // 2 - crop_r // 2
    sc = size_c // 2 - crop_c // 2
    if n_dim == 2:
        return vol[sr:sr+crop_r,sc:sc+crop_c]
    else:
        return vol[sz:sz+crop_z,sr:sr+crop_r,sc:sc+crop_c]


def bbox_for_mask(m : NDArray) -> typing.Union[BBox2D, BBox3D]:
    """calculate bounding box encompassing 'on' pixels of mask

    :param m: input mask
    :type m: NDArray
    :returns: bounding box for mask
    :rtype: Union[BBox2D, BBox3D]
    """
    assert len(m.shape) == 2 or len(m.shape) == 3, \
        "input mask must be 2- or 3-D"
    xp = cupy.get_array_module(m)
    if len(m.shape) == 3:
        z, y, x = xp.where(m)
        lz, uz = xp.amin(z), xp.amax(z)
        ly, uy = xp.amin(y), xp.amax(y)
        lx, ux = xp.amin(x), xp.amax(x)
        return (lz, uz), (ly, uy), (lx, ux)
    else:
        y, x = xp.where(m)
        ly, uy = xp.amin(y), xp.amax(y)
        lx, ux = xp.amin(x), xp.amax(x)
        return (ly, uy), (lx, ux)


def shared_bbox_from_proj_threshold(
        v1 : NDArray, v2 : NDArray,
        proj_fun : str='max', thresh_val : float=0) -> BBox3D:
    """determine region shared by both volumes by thresholding
        'shared' means that both have data in the region

    :param v1: first volume
    :type v1: NDArray
    :param v2: second volume
    :type v2: NDArray
    :param proj_fun: function to project with
       one of ('max', 'mean', 'median')
    :type proj_fun: str
    :param thresh_val: value to threshold with to make mask
    :type thresh_val: float
    :returns: 3d bounding box of region common to `v1` and `v2`
    :rtype: BBox3D
    """
    assert proj_fun in ('max', 'mean', 'median'), \
        "invalid `proj_fun` must be one of ('max', 'mean', 'median')"
    xp = cupy.get_array_module(v1)
    if proj_fun == 'max':
        fun = xp.amax
    elif proj_fun == 'mean':
        fun = xp.mean
    else:
        fun = 'median'
    bb_yx = bbox_for_mask(
        xp.logical_and(fun(v1, 0)>thresh_val, fun(v2, 0)>thresh_val)
    )
    bb_yx = [0, xp.inf] + list(bb_yx[0]) + list(bb_yx[1])
    bb_zx = bbox_for_mask(
        xp.logical_and(fun(v1, 1)>thresh_val, fun(v2, 1)>thresh_val)
    )
    bb_zx = list(bb_zx[0]) + [0, xp.inf] + list(bb_zx[1])
    bb_zy = bbox_for_mask(
        xp.logical_and(fun(v1, 2)>thresh_val, fun(v2, 2)>thresh_val)
    )
    bb_zy = list(bb_zy[0]) + list(bb_zy[1]) + [0, xp.inf]
    bb = numpy.vstack([bb_yx, bb_zx, bb_zy])
    starts = xp.amax(bb, 0)[::2].astype(int)
    ends = xp.amin(bb, 0)[1::2].astype(int)
    bb = [(int(starts[i]), int(ends[i])) for i in range(3)]
    return tuple(bb)


## texture utilities
def create_texture_object(data : NDArray,
                          address_mode: str,
                          filter_mode: str,
                          read_mode: str):
    """create_texture_object Make a texture object from input array.

    Args:
        data (NDArray): array to copy into texture memory
        address_mode (str): one of ('wrap', 'clamp', 'mirror', 'border')
        filter_mode (str): one of ('nearest', 'linear')
        read_mode (str): one of ('element_type', 'normalized_float')

    Returns:
        (cupy.cuda.texture.TextureObject, cupy.cuda.texture.CUDAarray): tuple of the texture object and CUDAarray
    """
    if cupy.issubdtype(data.dtype, cupy.unsignedinteger):
        fmt_kind = runtime.cudaChannelFormatKindUnsigned
    elif cupy.issubdtype(data.dtype, cupy.integer):
        fmt_kind = runtime.cudaChannelFormatKindSigned
    elif cupy.issubdtype(data.dtype, cupy.floating):
        fmt_kind = runtime.cudaChannelFormatKindFloat
    else:
        raise ValueError('Unsupported data type')
    if address_mode == 'wrap':
        address_mode = runtime.cudaAddressModeWrap
    elif address_mode == 'clamp':
        address_mode = runtime.cudaAddressModeClamp
    elif address_mode == 'mirror':
        address_mode = runtime.cudaAddressModeMirror
    elif address_mode == 'border':
        address_mode = runtime.cudaAddressModeBorder
    else:
        raise ValueError(
            'Unsupported address mode '
            '(supported: wrap, clamp, mirror, border)')
    if filter_mode == 'nearest':
        filter_mode = runtime.cudaFilterModePoint
    elif filter_mode == 'linear':
        filter_mode = runtime.cudaFilterModeLinear
    else:
        raise ValueError(
            'Unsupported filter mode (supported: nearest, linear)')
    if read_mode == 'element_type':
        read_mode = runtime.cudaReadModeElementType
    elif read_mode == 'normalized_float':
        read_mode = runtime.cudaReadModeNormalizedFloat
    else:
        raise ValueError(
            'Unsupported read mode '
            '(supported: element_type, normalized_float)')
    texture_fmt = texture.ChannelFormatDescriptor(
        data.itemsize * 8, 0, 0, 0, fmt_kind)
    array = texture.CUDAarray(texture_fmt, *data.shape[::-1])
    res_desc = texture.ResourceDescriptor(
        runtime.cudaResourceTypeArray, cuArr=array)
    tex_desc = texture.TextureDescriptor(
        (address_mode, ) * data.ndim, filter_mode, read_mode)
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    array.copy_from(data)
    return tex_obj, array


## image processing utilities
def threshold_triangle(im : NDArray, nbins : int=256) -> float:
    """threshold_triangle Compute threshold value by triangle method. Copied from `skimage.filters` but modified so that numpy or cupy is used based on input device.

    Args:
        im (NDArray): array to calculate threshold for
        nbins (int, optional): number of bins to use in histogram. Defaults to 256.

    Returns:
        float
    """
    xp = cupy.get_array_module(im)
    
    hist, bin_edges = xp.histogram(im, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    nbins = len(hist)

    # Find peak, lowest and highest gray levels.
    arg_pk_hgt = xp.argmax(hist)
    peak_height = hist[arg_pk_hgt]
    arg_llev, arg_hlev = xp.flatnonzero(hist)[[0, -1]]

    # Flip is True if left tail is shorter.
    flip = arg_pk_hgt - arg_llev < arg_hlev - arg_pk_hgt
    if flip:
        hist = hist[::-1]
        arg_llev = nbins - arg_hlev - 1
        arg_pk_hgt = nbins - arg_pk_hgt - 1

    # If flip == True, arg_hlev becomes incorrect
    # but we don't need it anymore.
    del(arg_hlev)

    # set up coordinate system
    width = arg_pk_hgt - arg_llev
    x1 = xp.arange(width)
    y1 = hist[x1 + arg_llev]

    # normalize
    norm = int(xp.sqrt(peak_height**2 + width**2))
    peak_height = peak_height / norm
    width = width / norm

    # maximize the length
    length = peak_height * x1 - width * y1
    arg_level = xp.argmax(length) + arg_llev

    if flip:
        arg_level = nbins - arg_level - 1
    return bin_centers[arg_level]


def _cuda_gridsize_for_blocksize(dim : int, block_size : int) -> int:
    return (dim + block_size - 1) // block_size


def launch_params_for_volume(shp : Iterable[int], 
                             block_size_z : int, 
                             block_size_r : int,
                             block_size_c : int) -> CuLaunchParameters:
    """launch_params_for_volume Automatically calculate good launch parameters for CUDA kernel that will be run on a volume of specified `shape`.

    Args:
        shp (Iterable[int]): shape of input volume kernel will be run over (ZRC).
        block_size_z (int): block size in z dimension (0 axis)
        block_size_r (int): block size in r dimension (1 axis)
        block_size_c (int): block size in c dimension (2 axis)

    Returns:
        CuLaunchParameters
    """
    #gz = _cuda_gridsize_for_blocksize(shp[0], block_size_z)
    #gr = _cuda_gridsize_for_blocksize(shp[1], block_size_r)
    #gc = _cuda_gridsize_for_blocksize(shp[2], block_size_c)
     
    #return (gz, gr, gc), (block_size_z, block_size_r, block_size_c)
    return (8, 8, 8), (block_size_z, block_size_r, block_size_c)

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.integer):
            return int(obj)
        return super().default(obj)
    

def uint16_to_uint8(a : NDArray, max_val : float = 65535) -> NDArray:
    """uint16_to_uint8 Convert input of type uint16 to uint8.

    Args:
        a (NDArray): input volume
        max_val (float, optional): maximum value of input array. Defaults to 65535.

    Returns:
        NDArray
    """
    xp = cupy.get_array_module(a)
    return xp.clip(xp.round(a*(255.0/max_val)), 0, 255).astype(xp.uint8)