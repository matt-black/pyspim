import cupy as cp
from .._util import create_texture_object


## CORRELATION RATIO
corr_kernel_source=r'''
extern "C"{
__global__ void corrKernel(double *sqrsum, double *corsum,
                           double *d_aff, float* target, cudaTextureObject_t src_tex,
                           size_t sx, size_t sy, size_t sz, size_t sx2, size_t sy2, size_t sz2) {
        const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
	const size_t y = blockDim.y * blockIdx.y + threadIdx.y;

        size_t z;
	float t, s;
	double ss = 0, st = 0;
	// coordinates transformation
	if (x < sx && y < sy)
        {
		for (z = 0; z < sz; z++)
                {
			float ix = (float)x;
			float iy = (float)y;
			float iz = (float)z;
			float tx = d_aff[0] * ix + d_aff[1] * iy + d_aff[2] * iz + d_aff[3] + 0.5;
			float ty = d_aff[4] * ix + d_aff[5] * iy + d_aff[6] * iz + d_aff[7] + 0.5;
			float tz = d_aff[8] * ix + d_aff[9] * iy + d_aff[10] * iz + d_aff[11] + 0.5;
			if (tx>0 && tx < sx2 && ty>0 && ty < sy2 && tz>0 && tz < sz2)
				s = tex3D<float>(src_tex, tx, ty, tz);
			else
				s = 0;
			t = target[x + y*sx + z*sx*sy];
			ss += (double)s*s;
			st += (double)s*t;
		}
		sqrsum[x + y*sx] = ss;
		corsum[x + y*sx] = st;
	}
}
}
'''


def _corr_ratio_reftex_raw(texA, imB, stdB, tM, zdimA, ydimA, xdimA, block_size=4):
    zdimB, ydimB, xdimB = imB.shape
    imB = imB.astype(cp.float32, copy=False)
    # preallocate output
    sqrsum = cp.zeros((xdimA*ydimA,), dtype=cp.uint64)
    corsum = cp.zeros_like(sqrsum)
    # make the kernel
    ckern = cp.RawKernel(corr_kernel_source, 'corrKernel')
    # figure out computation grid
    grid_x = (xdimB + block_size - 1) // block_size
    grid_y = (ydimB + block_size - 1) // block_size
    # call the kernel
    ckern((grid_x, grid_y), (block_size, block_size),
          (sqrsum, corsum, tM, imB, texA,
           xdimB, ydimB, zdimB, xdimA, ydimA, zdimA))
    # finish the calculation
    sqrsum = cp.sum(sqrsum)
    corsum = cp.sum(corsum)
    if cp.sqrt(sqrsum) == 0:
        return -2.0
    else:
        return float(cp.sqrt(corsum)/(cp.sqrt(sqrsum)*stdB))
    

def _corr_ratio_refcpy_raw(imA, imB, tM, block_size=4):
    """compute the correlation ratio using the CUDA kernel from [1]

    for how the CUDA setup works, see [2]
    
    References
    ---
    [1] Guo et al. "Rapid image deconvolution...", doi:10.1038/s41587-020-0560-x
    [2] https://github.com/cupy/cupy/pull/2432
    """
    zda, yda, xda = imA.shape
    imA = imA.astype(cp.float32)
    stdA = cp.sqrt(cp.sum(imA))
    imB = imB.astype(cp.float32)
    stdB = cp.sqrt(cp.sum(imB))
    # need to move the reference image (A) to texture
    tex_obj, tex_arr = create_texture_object(imA, 'border', 'linear',
                                             'element_type')
    cr = correlation_ratio_reftex(tex_obj, imB, stdB, tM, zda, yda, xda,
                                  block_size)
    del tex_obj, tex_arr
    return cr


_corr_ratio_sum_kernel = cp.ElementwiseKernel(
    'U texObj, raw T target, raw float32 m, uint64 height, uint64 width',
    'T corsum, T sqrsum',
    '''
    float4 voxel = make_float4(
        (float)(i / (width * height)) + .5f,
        (float)((i % (width * height)) / width) + .5f,
        (float)((i % (width * height)) % width) + .5f,
        1.0f
    );

    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3]));
    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7]));
    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11]));

    T s = tex3D<T>(texObj, z, y, x);
    corsum = s * target[i];
    sqrsum = s * s;
    ''',
    'correlation_ratio',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ''')


def _corr_ratio_refcpy_ewk(imA, imB, tM):
    tex_obj, tex_arr = create_texture_object(imA, 'border', 'linear',
                                             'element_type')
    sqrsum = cp.empty_like(imB)
    corsum = cp.empty_like(imA)
    _corr_ratio_sum_kernel(tex_obj, imB, tM.astype(cp.float32), *imB.shape[1:],
                           corsum, sqrsum)
    del tex_obj, tex_arr
    return cp.sum(corsum) / cp.sum(sqrsum)


def _corr_ratio_reftex_ewk(imA, imB, tM):
    sqrsum, corsum = cp.empty_like(imB), cp.empty_like(imB)
    tM = cp.asarray(tM).astype(cp.float32)
    _corr_ratio_sum_kernel(imA, imB, tM, *imB.shape[1:],
                           corsum, sqrsum)
    return cp.sum(corsum) / cp.sum(sqrsum)


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
_joint_hist_preproc_kernel = cp.ElementwiseKernel(
    'U texObj, raw T target, raw float32 m, T b1, T b2, uint64 height, uint64 width',
    'T J1, T Jx',
    '''
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
    ''',
    'histogram_prep',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ''')


@cp.fuse()
def discrete_entropy(hist_vals):
    """compute the (unnormalized) entropy of the input
    histogram values, as if they're a probability distribution

    as defined here, the entropy will scale like log2(# bins)
    so to normalize it, take the output of this and divide by that
    """
    return -cp.sum(hist_vals * cp.log2(hist_vals))


def _preprocess_image_pair_reftex(ref, tar, tM, nbin_ref, nbin_tar):
    tar = tar.astype(cp.float32, copy=False)
    tM = cp.asarray(tM).astype(cp.float32)
    j1, jx = cp.empty_like(tar), cp.empty_like(tar)
    _joint_hist_preproc_kernel(ref, tar, tM, nbin_ref, nbin_tar,
                               *tar.shape[1:], j1, jx)
    return j1, jx


def _preprocess_image_pair_refcpy(ref, tar, tM, nbin_ref, nbin_tar):
    ref_tobj, ref_tarr = create_texture_object(ref, 'border', 'linear',
                                               'element_type')
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM,
                                           nbin_ref, nbin_tar)
    del ref_tobj, ref_tarr
    return j1, jx


def _mutual_information_precomp_target(ref_tobj, tar, tM,
                                       nbin_ref, nbin_tar, H_tar):
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM,
                                           nbin_ref, nbin_tar)
    # calculate 1d histogram for reference image
    P_ref, _ = cp.histogram(j1, nbin_ref)
    P_ref = P_ref / cp.sum(P_ref) + cp.finfo(float).eps
    H_ref = discrete_entropy(P_ref) / cp.log2(nbin_ref)
    # calculate 1d histogram for proxy image
    P_rt2, _ = cp.histogram(jx, nbin_ref*nbin_tar)
    P_rt2 = P_rt2 / cp.sum(P_rt2) + cp.finfo(float).eps
    H_rt2 = discrete_entropy(P_rt2) / cp.log2(nbin_ref*nbin_tar)
    return H_tar + H_ref - H_rt2


def mutual_information(ref, tar, tM, nbin_ref, nbin_tar):
    """compute mutual information between input images `ref` & `tar`
    """
    ref_tex, tex_arr = create_texture_object(ref, 'border', 'linear',
                                             'element_type')
    j1, jx = _preprocess_image_pair_reftex(
        ref_tex, tar, tM, nbin_ref, nbin_tar
    )
    # compute entropy of target
    P_tar, _ = cp.histogram(tar, nbin_tar)
    P_tar = P_tar / cp.sum(P_tar) + cp.finfo(float).eps
    H_tar = discrete_entropy(P_tar) / cp.log2(nbin_tar)
    mi = mutual_information_precomp_target(ref_tex, tar, cp.eye(4),
                                           nbin_ref, nbin_tar, H_tar)
    del ref_tex, tex_arr
    return mi
    

def _entropy_correlation_coeff_precomp_target(ref_tobj, tar, tM,
                                              nbin_ref, nbin_tar,
                                              H_tar):
    j1, jx = _preprocess_image_pair_reftex(ref_tobj, tar, tM,
                                           nbin_ref, nbin_tar)
    # calculate 1d histogram for reference image
    P_ref, _ = cp.histogram(j1, nbin_ref)
    P_ref = P_ref / cp.sum(P_ref) + cp.finfo(float).eps
    H_ref = discrete_entropy(P_ref) / cp.log2(nbin_ref)
    # calculate 1d histogram for proxy image
    P_rt2, _ = cp.histogram(jx, nbin_ref*nbin_tar)
    P_rt2 = P_rt2 / cp.sum(P_rt2) + cp.finfo(float).eps
    H_rt2 = discrete_entropy(P_rt2) / cp.log2(nbin_ref*nbin_tar)
    return 2 - 2 * (H_rt2 / (H_ref + H_tar))


def entropy_correlation_coeff(ref, tar, tM, nbin_ref, nbin_tar):
    """
    """
    ref_tex, tex_arr = create_texture_object(ref, 'border', 'linear',
                                             'element_type')
    j1, jx = _preprocess_image_pair_reftex(
        ref_tex, tar, tM, nbin_ref, nbin_tar
    )
    # compute entropy of target
    P_tar, _ = cp.histogram(tar, nbin_tar)
    P_tar = P_tar / cp.sum(P_tar) + cp.finfo(float).eps
    H_tar = discrete_entropy(P_tar) / cp.log2(nbin_tar)
    ec = entropy_correlation_coeff_precomp_target(
        ref_tex, tar, cp.eye(4), nbin_ref, nbin_tar, H_tar
    )
    del ref_tex, tex_arr
    return ec
