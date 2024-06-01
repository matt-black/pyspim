"""cupy ElementwiseKernel to combine two images, one of which (target) is
registered into the view of the other (passed in as a texture object, `texObj`)
by some transform `m`
"""
_combine_img_kernel = cupy.ElementwiseKernel(
    'U texObj, raw T target, raw float32 m, uint64 height, uint64 width',
    'T comb',
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

    T comb = (tex3D<T>(texObj, z, y, x) + target[i]) / 2.0;
    
    ''',
    'combine_imgs_with_transform',
    preamble='''
    inline __host__ __device__ float dot(float4 a, float4 b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    '''
)
