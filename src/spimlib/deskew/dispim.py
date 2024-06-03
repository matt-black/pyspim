"""
"""
import cupy

from .typing import NDArray
from .._util import create_texture_object


def deskew_stage_scan(im : NDArray, pixel_size : float, step_size : float,
                      direction : int, **kwargs) -> NDArray:
    """deskew stage-scanned volume into "normal" diSPIM coordinate system
        (see [1], esp. Figure 1)
        this function will dispatch to a CUDA-kernel based implementation that
        mimics that of the original Shroff lab Fiji plugin

    References
    ---
    [1]  Kumar, et. al, "Using Stage- and Slit-...", doi: 10.1086/689589

    :param im: input volume
    :type im: NDArray
    :param pixel_size: pixel size, in real (space) units
    :type pixel_size: float
    :param step_size: step size, in real (space) units
    :type step_size: float
    :param direction: 
    :type direction: int
    :returns: deskewed volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(im)
    if xp == cupy:
        return _deskew_stage_scan_gpu(im, pixel_size, step_size, direction,
                                      **kwargs)
    else:
        return _deskew_stage_scan_cpu(im, pixel_size, step_size, direction)


def _deskew_stage_scan_gpu(im : NDArray, pixel_size : float, step_size : float,
                           direction : int, block_size : int=4) -> NDArray:
    assert direction == 1 or direction == -1, \
        "direction must be +/- 1"
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
    tex_obj, tex_arr = create_texture_object(im, 'border', 'nearest',
                                             'element_type')
    depth, height, width = im.shape
    # need to rescale width s.t. entire original image
    # will fit in the deskewed output
    width = width + round(depth*abs(step_size/pixel_size))
    out = cupy.zeros((depth, height, width), dtype=cp.float32)
    # setup the kernel
    dkern = cupy.RawKernel(__deskew_kernel_source, 'deskewKernel')
    # launch kernel
    grid_x = (width + block_size - 1) // block_size
    grid_y = (height + block_size - 1) // block_size
    dkern(
        (grid_x, grid_y), (block_size, block_size),
        (out, tex_obj, width, height, depth, pixel_size/step_size)
    )
    del tex_obj, tex_arr
    # need to flip back to original orientation (see above)
    if direction < 0:
        out = cupy.flip(out, -1)
        im  = cupy.flip(im, -1)
    # return the same data type as the input image (probably uint16)
    # NOTE: this seems to solve a CUDAIllegalMemoryAccess error that
    # can occur if the output is left as a 32-bit float
    return out.astype(in_type)


__deskew_kernel_source=r'''
extern "C"{
__global__ void deskewKernel(float* deskewed, cudaTextureObject_t texObj,
                             int width, int height, int depth, double rel_pixel_size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    int z;
    if (x < width && y < height) {
        for (z = 0; z < depth; z++) {
            float x2 = (float)x * rel_pixel_size - z;
            deskewed[x + width*y + z*width*height] = tex3D<float>(texObj, x2/rel_pixel_size, y, z);
        }
    }
}
}
'''


def _deskew_stage_scan_cpu(im, pixel_size, step_size, direction):
    raise NotImplementedError('only GPU-based deskewing avail. right now')
