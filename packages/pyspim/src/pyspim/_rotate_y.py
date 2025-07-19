""" """


import cupy

from .typing import NDArray


def rotate_view(vol: NDArray, rot_pos: bool) -> NDArray:
    """rotate volume by +/- 90 degrees along Y-axis

    :param vol: volume to rotate
    :type vol: NDArray
    :param rot_pos: rotate in positive (`True`) or negative (`False`) direction
    :type rot_pos: bool
    :returns: rotated volume
    :rtype: NDArray
    """
    return rotate_by_kernel(vol, rot_pos)


def rotate_by_kernel(vol: NDArray, rot_pos: bool, block_size: int = 4) -> NDArray:
    """rotate volume by +/- 90 degrees along Y-axis using CUDA kernel
        kernel taken from microImageLib ([1])

    References
    ---
    [1] https://github.com/eguomin/microImageLib

    :param vol: volume to rotate
    :type vol: NDArray
    :param rot_pos: rotate in positive (`True`) or negative (`False`) direction
    :type rot_pos: bool
    :param block_size: block size to use for CUDA kernel
    :type block_size: int
    :returns: rotated volume
    :rtype: NDArray
    """
    assert len(vol.shape) == 3, "input must be a 3D volume"
    depth, height, width = vol.shape
    rot_dir = 1 if rot_pos else -1
    input_cpu = cupy.get_array_module(vol) != cupy
    if input_cpu:
        vol = cupy.asarray(vol)
    if vol.dtype == cupy.float32:
        rkern = cupy.RawKernel(__rotation_kernel_source, "rotYkernelFloat")
    else:
        rkern = cupy.RawKernel(__rotation_kernel_source, "rotYkernelUshort")
    # preallocate output
    out = cupy.zeros((width, height, depth), dtype=vol.dtype)
    # launch kernel
    gx = (width + block_size - 1) // block_size
    gy = (height + block_size - 1) // block_size
    gz = (depth + block_size - 1) // block_size
    rkern(
        (gx, gy, gz),
        (block_size, block_size, block_size),
        (out, vol, width, height, depth, rot_dir),
    )
    if input_cpu:
        return out.get()
    else:
        return out


__rotation_kernel_source = r"""
extern "C"{

__global__ void rotYkernelUshort(unsigned short *d_odata, unsigned short *d_idata,
                                 size_t sx, size_t sy, size_t sz, int rotDirection){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		if (rotDirection == 1)  //rotate 90 deg around Y axis 
			d_odata[(sx - i - 1)*sy*sz + j * sz + k] = d_idata[k*sy*sx + j * sx + i];
		else if(rotDirection == -1)  //rotate -90 deg around Y axis
			d_odata[i*sy*sz + j * sz + (sz - k - 1)] = d_idata[k*sy*sx + j * sx + i];
			
	}
}

__global__ void rotYkernelFloat(float *d_odata, float *d_idata,
                                 size_t sx, size_t sy, size_t sz, int rotDirection){
	const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
	const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
	if (i < sx && j < sy && k < sz){
		if (rotDirection == 1)  //rotate 90 deg around Y axis 
			d_odata[(sx - i - 1)*sy*sz + j * sz + k] = d_idata[k*sy*sx + j * sx + i];
		else if(rotDirection == -1)  //rotate -90 deg around Y axis
			d_odata[i*sy*sz + j * sz + (sz - k - 1)] = d_idata[k*sy*sx + j * sx + i];
			
	}
}

}
"""
