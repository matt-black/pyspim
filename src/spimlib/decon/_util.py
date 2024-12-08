import itertools
from typing import Optional, Tuple

import cupy
import numpy

from ..typing import NDArray, BBox3D, SliceWindow

__tup3 = Tuple[int,int,int]  # convenience type for output


def gaussian_kernel_1d(sigma : float, radius : int):
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius+1)
    phi = numpy.exp(-0.5 / sigma2 * x**2)
    return (phi / phi.sum()).astype(numpy.float32)


def crop_and_pad_for_deconv(vol : NDArray,
                            bbox : Optional[BBox3D], pad : int) -> NDArray:
    """crop and pad input volume in preparation for deconvolution
        deconvolution will produce artifacts at the edges of images (volumes)
        to help remedy this, the input can be padded, deconvolved, and then
        the padded regions cropped out post-deconvolution.
        this function will take the input volume, crop it, and then pad the
        edges such that after deconvolution, one can re-crop it to get
        rid of the aforementioned artifacts. instead of just cropping and
        padding, if the bbox and padding are still contained in the original
        volume, this function will under-crop the appropriate amount such that
        original parts of the volume are kept (instead of padding)
    
    :param vol: input volume to be deconvolved
    :type vol: NDArray
    :param bbox: bounding box to crop volume down to
        if `None`, cropping is not done
    :type bbox: Optional[BBox3D]
    :param pad: amount of padding past bbox (same for all axes)
    :type pad: int
    :returns: (under)cropped and possibly-padded volume ready for deconvolution
    :rtype: NDArray
    """
    if bbox is None:
        bbox = tuple(zip([0,0,0], list(vol.shape)))
    (padl, padr), (lbd, ubd) = _crop_bounds_and_padding(vol, bbox, pad)
    return _crop_and_pad(vol, padl, padr, lbd, ubd)



def _crop_bounds_and_padding(vol : NDArray, bbox : BBox3D, pad : int) -> \
        Tuple[Tuple[__tup3,__tup3], Tuple[__tup3, __tup3]]:
    """determine amount of padding and upper/lower bounds of crop box to
        use when cropping/padding the input volume in preparation
        for deconvolution

    :param vol: input volume
    :type vol: NDArray
    :param bbox: bounding box to crop volume down to
    :type bbox: BBox3D
    :param pad: amount of padding past bbox (same for all axes)
    :type pad: int
    :returns: paddings and upper/lower bounds
    """
    shp = vol.shape
    # figure out lower bound of (maybe) padded image
    lbd = [x[0] - pad for x, s in zip(bbox, shp)]
    # if the bound is negative, we have to pad to the left
    # otherwise, no padding
    pdl = tuple([abs(l) if l < 0 else 0 for l in lbd])
    # correct for negative bounds which are now 0 (b/c of padding)
    lbd = tuple([0 if l < 0 else l for l in lbd])
    ## now we follow the same logic for the upper bound except
    ## minuses are pluses and we have to check >shape instead of <0
    ubd = [x[1] + pad for x, _ in zip(bbox, shp)]
    pdr = tuple([v - s if v > s else 0 for v, s in zip(ubd, shp)])
    ubd = tuple([s if v > s else v for v, s in zip(ubd, shp)])
    return (pdl, pdr), (lbd, ubd)


def _crop_and_pad(vol : NDArray,
                  padl : __tup3, padr : __tup3,
                  lowb : __tup3, uppb : __tup3):
    """crop and pad the input volume w. specified parameters
        padding is done with option 'symmetric' for (see : `numpy.pad`)

    :param vol: input volume to be deconvolved
    :type vol: NDArray
    :param padl: left paddings for each axis
    :type padl: Tuple[int,int,int]
    :param padr: right paddings for each axis
    :type padr: Tuple[int,int,int]
    :param lowb: lower crop bounds for each axis
    :type lowb: Tuple[int,int,int]
    :param uppb: upper crop bounds for each axis
    :type uppb: Tuple[int,int,int]
    :returns: cropped and padded volume suitable for deconvolution
    :rtype: NDArray
    """
    xp = cupy.get_array_module(vol)
    return xp.pad(vol[lowb[0]:uppb[0],lowb[1]:uppb[1],lowb[2]:uppb[2]],
                  [(l, r) for l, r in zip(padl, padr)],
                  'symmetric')


## stable division: only divide `a / b` if denominator, `b`, isn't too small
div_stable = cupy.ElementwiseKernel(
    'T a, T b, float32 eps',
    'T o',
    '''
    o = (b > eps) ? a / b : 0;
    ''',
    'div_stable_kernel',
)


def initialize_estimate(a : cupy.ndarray, b : cupy.ndarray, order : str = 'F'):
    out = cupy.zeros(a.shape, dtype=cupy.float32, order=order)
    _initialize_estimate_kernel(a, b, out)
    return out


_initialize_estimate_kernel = cupy.ElementwiseKernel(
    'T a, T b', 'float32 o',
    '''
    o = (a + b) / 2.0f
    ''',
    'initialize_estimate_kernel'
)

## chunking

def _pad_amount(dim : int, chunk_dim : int) -> int:
    assert dim >= chunk_dim, \
        "dim : {:d}, chunk_dim : {:d}".format(dim, chunk_dim)
    n = 1
    while chunk_dim * n < dim:
        n += 1
    return chunk_dim * n - dim


def _pad_splits(pad_size : int) -> Tuple[int,int]:
    left = pad_size // 2
    right = pad_size // 2 + pad_size % 2
    return left, right


_chunk_pads = Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]] | \
    Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int],Tuple[int,int]]

class ChunkProps(object):
    def __init__(self, data_window : SliceWindow, 
                 read_window : SliceWindow,
                 paddings : _chunk_pads,
                 out_window : SliceWindow):
        self.data_window = data_window
        self.read_window = read_window
        self.paddings = paddings
        self.out_window = out_window


def calculate_decon_chunks(
    z : int, r : int, c : int,
    chunk_shape : int|Tuple[int,int,int],
    overlap : int|Tuple[int,int,int],
    channel_slice : slice|None,
) -> dict[int,ChunkProps]:
    shape = tuple([z, r, c])
    if isinstance(chunk_shape, int):
        chunk_shape = tuple([chunk_shape,]*3)
    else:
        chunk_shape = chunk_shape
    if isinstance(overlap, int):
        overlap = tuple([overlap,]*3)
    else:
        overlap = overlap
    # determine padding & resulting shape
    pad_size = [_pad_amount(d, cd) for d, cd in zip(shape, chunk_shape)]
    pads = [_pad_splits(p) for p in pad_size]
    padded_shape = [s+p for s, p in zip(shape, pad_size)]
    # determine how (padded) array will be chunked
    n_chunk = [s//c for s, c in zip(padded_shape, chunk_shape)]
    # make sure division worked correctly
    # (chunks should divide padded shape evenly)
    assert all([n*c==s for n, c, s 
                in zip(n_chunk, chunk_shape, padded_shape)]), \
        "padding didnt ensure correct division"
    chunk_mults = itertools.product(*[range(n) for n in n_chunk])
    chunk_windows = {}
    for chunk_idx, chunk_mult in enumerate(chunk_mults):
        # these indices represent where in the padded data we are reading from
        idx0 = [m * s for m, s in zip(chunk_mult, chunk_shape)]
        idx1 = [i0 + s for i0, s in zip(idx0, chunk_shape)]
        pad_idxs = list(zip(idx0, idx1))
        # now figure out where in the actual data this corresponds to
        data_idxs, pad_amts, read_idxs, out_idxs = [], [], [], []
        for dim_idx, (i0, i1) in enumerate(pad_idxs):
            i0d, i1d = i0 - pads[dim_idx][0], i1 - pads[dim_idx][0]
            # figure out conditions for "left" index
            if i0d < 0:  # we're not starting at data, in the "pad"
                left_data, left_read = 0, 0
                left_pad = pads[dim_idx][0]
                left_out = pads[dim_idx][0]
            else:  # i0d >= 0 -- we're in the data
                left_data = i0d
                left_read = max(i0d - overlap[dim_idx], 0)
                if left_read == 0:  # cant read full overlap in the data
                    left_pad = pads[dim_idx][0] - (overlap[dim_idx] - i0d)
                else:  # overlap region fully contained in data
                    left_pad = 0
                left_out = (left_data - left_read)+left_pad
            # figure out conditions for "right" index
            if i1d > shape[dim_idx]:  # we're outside of the data on the rhs
                right_data, right_read = shape[dim_idx], shape[dim_idx]
                right_pad = i1 - padded_shape[dim_idx]
            else:
                right_data = i1d
                right_read = right_data + overlap[dim_idx]
                if right_read > shape[dim_idx]:
                    over_size  = right_read - shape[dim_idx]
                    right_read = shape[dim_idx]
                    right_pad  = overlap[dim_idx] - over_size
                else:
                    right_pad = 0
            right_out = left_out + (right_data-left_data)
            data_idxs.append((left_data, right_data))
            pad_amts.append((left_pad, right_pad))
            read_idxs.append((left_read, right_read))
            out_idxs.append((left_out, right_out))
        # now make slicing windows for this chunk
        # data windows corresp. to the actual data that this chunk is
        # responsible for
        # read windows corresp. to what should actually get read in
        # when processing this chunk (includes overlap)
        data_window = [slice(left, right) for left, right in data_idxs]
        read_window = [slice(left, right) for left, right in read_idxs]
        out_window  = [slice(left, right) for left, right in out_idxs]
        if channel_slice is not None:
            data_window.insert(0, channel_slice)
            read_window.insert(0, channel_slice)
            out_window.insert(0, channel_slice)
            pad_amts.insert(0, (0,0))
        chunk_windows[chunk_idx] = ChunkProps(
            tuple(data_window), tuple(read_window),
            pad_amts, tuple(out_window)
        )
    return chunk_windows