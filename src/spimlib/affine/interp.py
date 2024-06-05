"""affine transformation utilities
"""
import cupy
import numpy
from scipy.ndimage import affine_transform as affine_transform_cpu

from ..typing import NDArray
from .._matrix import *
from .._util import create_texture_object


## GENERIC AFFINE TRANSFORMATIONS
## transform input volumes by 3d affine transform matrix using texture memory
_affine_transform_kernel = cupy.ElementwiseKernel(
    'U texObj, raw float32 m, uint64 height, uint64 width',
    'T out',
    '''
    float4 voxel = make_float4(
        (float)(i / (width * height)),
        (float)((i % (width * height)) / width),
        (float)((i % (width * height)) % width),
        1.0f
    );
    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3])) + .5f;
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7])) + .5f;
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11])) + .5f;
    out = tex3D<T>(texObj, z, y, x);
    ''',
    'affine_transform',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ''')


__affine_rawKernel_source=r'''
extern "C"{
__global__ void affineKernel(float *out, cudaTextureObject_t tex,
                             double* T,
                             size_t sx, size_t sy, size_t sz,
                             size_t sx2, size_t sy2, size_t sz2){
    const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t z = blockDim.z * blockIdx.z + threadIdx.z;
    // coordinates transformation
    if (x < sx && y < sy && z < sz){
	float ix = (float)x;
	float iy = (float)y;
	float iz = (float)z;
	float tx = T[0] * ix + T[1] * iy + T[2] * iz + T[3]+0.5;
	float ty = T[4] * ix + T[5] * iy + T[6] * iz + T[7]+0.5;
	float tz = T[8] * ix + T[9] * iy + T[10] * iz + T[11]+0.5;
        // texture interpolation
	if (tx >= 0 && tx < sx2 && ty >= 0 && ty < sy2 && tz >= 0 && tz < sz2) {
            out[x + y*sx + z*sx*sy] = tex3D<float>(tex, tx, ty, tz);
	}
    }
}
}
'''


def transform(vol : NDArray, tM : NDArray) -> NDArray:
    """transform the input volume using the specified affine transform.
        uses linear interpolation to calculate values in transformed volume.

    :param vol: array of volumetric data to transform
    :type vol: NDArray
    :param tM: transform matrix
    :type tM: NDArray
    :returns: transformed volume
    :rtype: NDArray
    """
    xp = cupy.get_array_module(vol)
    in_type = vol.dtype
    tM = cupy.asarray(tM).astype(cupy.float32)
    # move to texture memory
    vol = vol.astype(xp.float32, copy=False)
    tex_obj, tex_arr = create_texture_object(vol, 'border', 'linear', 'element_type')
    out = cupy.zeros(vol.shape, dtype=cupy.float32)
    _affine_transform_kernel(tex_obj, tM, *vol.shape[1:], out)
    vol = vol.astype(in_type, copy=False)
    del tex_obj, tex_arr
    return out


def rotate_about_point(vol : NDArray,
                       x : float, y : float, z : float,
                       alpha : float, beta : float, gamma : float) -> NDArray:
    """rotate volume about the point `(x,y,z)`

    :param vol: input volume
    :type vol: NDArray
    :param x: x-coordinate of point to rotate about
    :type x: float
    :param y: y-coordinate of point to rotate about
    :type y: float
    :param z: z-coordinate of point to rotate about
    :type z: float
    :param alpha: rotation angle along x-axis
    :type alpha: float
    :param beta: rotation angle along y-axis
    :type beta: float
    :param gamma: rotation angle along z-axis
    :type gamma: float
    :returns: rotated volume
    :rtype: NDArray
    """
    t1 = translation_matrix(x, y, z)
    R  = rotation_matrix(alpha, beta, gamma)
    t2 = translation_matrix(-x, -y, -z)
    return transform(vol, t1 @ R @ t2)


def rotate_about_center(vol : NDArray,
                        alpha : float, beta : float, gamma : float) -> NDArray:
    """rotate volume about its center

    :param vol: input volume 
    :type vol: NDArray
    :param alpha: rotation angle along x-axis
    :type alpha: float
    :param beta: rotation angle along y-axis
    :type beta: float
    :param gamma: rotation angle along z-axis
    :type gamma: float
    :returns: rotated volume
    :rtype: NDArray
    """
    depth, height, width = vol.shape
    x = (width - 1) / 2.0
    y = (height - 1) / 2.0
    z = (depth - 1) / 2.0
    return rotate_about_point(vol, x, y, z, alpha, beta, gamma)


# BELOW: works, but is v. memory-intensive b/c of the padding, kept
# as possible reference for testing better memory functions since this
# is definitely accurate
# def rotate_volume(vol, angle, axis):
#     """rotate the input `vol` about its center

#     this is different than normal `ndimage.rotate` because that rotates
#     the volume about the plane defined by the specified axis
#     """
#     sp = get_scipy_module(vol)
#     axes = {'x' : (0,1), 'y' : (0,2), 'z' : (1,2)}[axis]
#     pads = [(vol.shape[i]-vol.shape[i]//2, vol.shape[i]//2)
#             for i in range(len(vol.shape))]
#     return sp.ndimage.rotate(
#         cupy.pad(vol, pads, 'constant'),
#         angle, axes=axes, reshape=False
#     )[pads[0][0]:-pads[0][1],pads[1][0]:-pads[1][1],pads[2][0]:-pads[2][1]]
