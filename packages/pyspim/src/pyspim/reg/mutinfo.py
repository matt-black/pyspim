import cupy

from .._util import create_texture_object
from ..typing import NDArray

## MUTUAL INFORMATION
# mutual information based metrics
"""cupy implementation of fast histogramming, as described in [1]

[1] R. Shams & N. Barnes, "Speeding up Mutual Information Computation Using NVIDIA CUDA Hardware"
    http://users.cecs.anu.edu.au/~ramtin/papers/2007/DICTA_2007a.pdf

kernel calculates J(x) as in eqn. (4) of [1]

inputs:
  texObj: the source (reference) image to be aligned to, in texture memory
  target: the target image to be aligned to source
  m : the candidate transformation matrix to align target & reference
  b1 : number of bins for source image
  b2 : number of bins for target image
  height : height (y-dim) of target
  width : width (x-dim) of target

outputs:
  J1 : the transformed source image (to be histogrammed)
  Jx : the combined intensity image
"""
_joint_hist_preproc_kernel = cupy.ElementwiseKernel(
    "U texObj, raw T target, raw float32 m, T b1, T b2, uint64 height, uint64 width",
    "T J1, T Jx",
    """
    float4 voxel = make_float4(
        (float)(i / (width * height)) + .5f,
        (float)((i % (width * height)) / width) + .5f,
        (float)((i % (width * height)) % width) + .5f,
        1.0f
    );

    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3]));
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7]));
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11]));

    J1 = tex3D<T>(texObj, z, y, x);
    Jx = (b1 * (J1 + target[i]*(b2-1))) / (b1*b2-1);
    """,
    "histogram_prep",
    preamble="""
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    """,
)


discrete_entropy = cupy.ReductionKernel(
    "T h", "T y", "h * log2(h)", "a + b", "y = -a", ""
)


def _preprocess_image_pair_reftex(
    ref, tar: NDArray, tM: NDArray, nbin_ref: int, nbin_tar: int
):
    tar = tar.astype(cupy.float32, copy=False)
    tM = cupy.asarray(tM).astype(cupy.float32)
    j1, jx = cupy.empty_like(tar), cupy.empty_like(tar)
    _joint_hist_preproc_kernel(ref, tar, tM, nbin_ref, nbin_tar, *tar.shape[1:], j1, jx)
    return j1, jx


def _preprocess_image_pair_refcpy(
    ref: NDArray, tar: NDArray, tM: NDArray, nbin_ref: int, nbin_tar: int
):
    ref_tobj, ref_tarr = create_texture_object(ref, "border", "linear", "element_type")
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM, nbin_ref, nbin_tar)
    del ref_tobj, ref_tarr
    return j1, jx


def _mutual_information_precomp_target(
    ref_tobj, tar: NDArray, tM: NDArray, nbin_ref: int, nbin_tar: int, H_tar: float
):
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM, nbin_ref, nbin_tar)
    # calculate 1d histogram for reference image
    P_ref, _ = cupy.histogram(j1, nbin_ref)
    P_ref = P_ref / cupy.sum(P_ref) + cupy.finfo(float).eps
    H_ref = discrete_entropy(P_ref) / cupy.log2(nbin_ref)
    # calculate 1d histogram for proxy image
    P_rt2, _ = cupy.histogram(jx, nbin_ref * nbin_tar)
    P_rt2 = P_rt2 / cupy.sum(P_rt2) + cupy.finfo(float).eps
    H_rt2 = discrete_entropy(P_rt2) / cupy.log2(nbin_ref * nbin_tar)
    return H_tar + H_ref - H_rt2


def mutual_information(
    ref: NDArray, tar: NDArray, tM: NDArray, nbin_ref: int, nbin_tar: int
) -> float:
    """mutual information between input images `ref` & `tar` where `tar`
        is first transformed by matrix `tM`

    :param ref: reference volume
    :type ref: NDArray
    :param tar: target (AKA "moving") volume
    :type tar: NDArray
    :param tM: matrix to transform `tar` with
    :type tM: NDArray
    :param nbin_ref: # of bins in histogram of `ref`
    :type nbin_ref: int
    :param nbin_tar: # of bins in histogram of `tar`
    :type nbin_tar: int
    :returns: mutual information between `ref` and transformed `tar`
    :rtype: float
    """
    ref_tex, tex_arr = create_texture_object(ref, "border", "linear", "element_type")
    # j1, jx = _preprocess_image_pair_reftex(
    #    ref_tex, tar, tM, nbin_ref, nbin_tar
    # )
    # compute entropy of target
    P_tar, _ = cupy.histogram(tar, nbin_tar)
    P_tar = P_tar / cupy.sum(P_tar) + cupy.finfo(float).eps
    H_tar = discrete_entropy(P_tar) / cupy.log2(nbin_tar)
    mi = _mutual_information_precomp_target(
        ref_tex, tar, cupy.eye(4), nbin_ref, nbin_tar, H_tar
    )
    del ref_tex, tex_arr
    return mi


def _entropy_correlation_coeff_precomp_target(
    ref_tobj, tar, tM, nbin_ref, nbin_tar, H_tar
):
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM, nbin_ref, nbin_tar)
    # calculate 1d histogram for reference image
    P_ref, _ = cupy.histogram(j1, nbin_ref)
    P_ref = P_ref / cupy.sum(P_ref) + cupy.finfo(float).eps
    H_ref = discrete_entropy(P_ref) / cupy.log2(nbin_ref)
    # calculate 1d histogram for proxy image
    P_rt2, _ = cupy.histogram(jx, nbin_ref * nbin_tar)
    P_rt2 = P_rt2 / cupy.sum(P_rt2) + cupy.finfo(float).eps
    H_rt2 = discrete_entropy(P_rt2) / cupy.log2(nbin_ref * nbin_tar)
    return 2 - 2 * (H_rt2 / (H_ref + H_tar))


def entropy_correlation_coeff(
    ref: NDArray, tar: NDArray, tM: NDArray, nbin_ref: int, nbin_tar: int
) -> float:
    """entropy correlation coefficient between input images `ref` & `tar`
        where `tar` is first transformed by matrix `tM`

    :param ref: reference volume
    :type ref: NDArray
    :param tar: target (AKA "moving") volume
    :type tar: NDArray
    :param tM: matrix to transform `tar` with
    :type tM: NDArray
    :param nbin_ref: # of bins in histogram of `ref`
    :type nbin_ref: int
    :param nbin_tar: # of bins in histogram of `tar`
    :type nbin_tar: int
    :returns: entropy correlation coefficient between `ref` and transformed `tar`
    :rtype: float
    """
    ref_tex, tex_arr = create_texture_object(ref, "border", "linear", "element_type")
    # j1, jx = _preprocess_image_pair_reftex(
    #    ref_tex, tar, tM, nbin_ref, nbin_tar
    # )
    # compute entropy of target
    P_tar, _ = cupy.histogram(tar, nbin_tar)
    P_tar = P_tar / cupy.sum(P_tar) + cupy.finfo(float).eps
    H_tar = discrete_entropy(P_tar) / cupy.log2(nbin_tar)
    ec = _entropy_correlation_coeff_precomp_target(
        ref_tex, tar, tM, nbin_ref, nbin_tar, H_tar
    )
    del ref_tex, tex_arr
    return ec
