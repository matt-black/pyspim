"""private utilities
"""
from collections.abc import Iterable
from functools import partial

import cupy
import numpy
from numpy.typing import ArrayLike
# skimage
import cucim
import skimage
# scipy
import scipy
import cupyx.scipy
import cupyx.scipy.fft as fftgpu
import scipy.fft as fftcpu
# cupy.cuda
from cupy.cuda import texture
from cupy.cuda import runtime

## utilities for determining CPU/GPU libraries from inputs
def get_skimage_module(arr_or_xp):
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return skimage
    else:
        return cucim.skimage


def get_fft_module(arr_or_xp):
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return fftcpu
    else:
        return fftgpu


def get_scipy_module(arr_or_xp):
    if isinstance(arr_or_xp, (numpy.ndarray, cupy.ndarray)):
        xp = cupy.get_array_module(arr_or_xp)
    else:
        xp = arr_or_xp
    if xp == numpy:
        return scipy
    else:
        return cupyx.scipy


## misc. support functions
def supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : cupy.dtype or Iterable of cupy.dtype
        The input dtype. If a sequence of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `cupy.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
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
        return cupy.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = cupy.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, cupy.float64)


## padding/shape utilities
def shape_for_divisible(shp, *div):
    """compute the shape >= input `shp` that is divisible in each
    dimension by the passed-in args, `*div`
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


def pad_to_shape(vol, shape, **kwargs):
    """pad input volume to specified shape by padding
    (as equally as possible) to edges s.t. original volume
    is centered in padded image
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


def unpad_to_shape(vol, shape):
    """get rid of padding due to `pad_to_shape`
    """
    s_, v_ = list(shape), []
    for i, s in enumerate(s_):
        v_.append(vol.shape[i])
    pads = [v-s for v, s in zip(v_, s_)]
    assert all([p>=0 for p in pads]), \
        "output shape must be lte than current in all dimensions"
    starts = [p//2 for p in pads]
    ends = [v-(p//2 + p%2) for v, p in zip(v_, pads)]
    slices = tuple([slice(s, e) for s, e in zip(starts, ends)])
    return vol[slices]


def pad_to_same_size(a, b, **kwargs):
    """pad input volumes `a` & `b` to the same size
    `**kwargs` are passed to `cp.pad`
    """
    za, ya, xa = a.shape
    zb, yb, xb = b.shape
    z, y, x = max([za, zb]), max([ya, yb]), max([xa, xb])
    fn = partial(pad_to_shape, shape=[z, y, x], **kwargs)
    return fn(a), fn(b)


def center_crop(vol : ArrayLike, *args):
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
        sz = size_z // 2 - crop_z // 2
    sr = size_r // 2 - crop_r // 2
    sc = size_c // 2 - crop_c // 2
    if n_dim == 2:
        return vol[sr:sr+crop_r,sc:sc+crop_c]
    else:
        return vol[sz:sz+crop_z,sr:sr+crop_r,sc:sc+crop_c]


## texture utilities
def create_texture_object(data,
                          address_mode: str,
                          filter_mode: str,
                          read_mode: str):
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
def threshold_triangle(im, nbins=256):
    """compute threshold value for `im` by triangle method

    copied from `skimage.filters` with histogram modified
    to use numpy/cupy
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
