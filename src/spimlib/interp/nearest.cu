inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z + b.z + a.w * b.w;
}

inline __host__ __device__ size_t xyz2idx(int x, int y, int z, 
                                          size_t width, size_t height)
{
    return x + ((z * height) + y) * width;
}

inline __host__ __device__ int nearestNeighbor(float coord)
{
    if (fmodf(coord, 1.0f) < 0.5f) {
        return __float2int_rd(coord);
    } else {
        return __float2int_rd(coord)+1;
    }
}

template<typename T>
__global__ void affineTransformNearest(float* out, T* in, float* M_aff,
                                       size_t sz_o, size_t sy_o, size_t sx_o,
                                       size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        float x_t = dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
        int x_i = nearestNeighbor(x_t);
        float y_t = dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
        int y_i = nearestNeighbor(y_t);
        float z_t = dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
        int z_i = nearestNeighbor(z_t);
        if (x_i >= 0 && x_i < sx_i-1 && y_i >= 0 && y_i < sy_i-1 && 
            z_i >= 0 && z_i < sz_i-1) {
            size_t idxi = xyz2idx(x_i, y_i, z_i, sx_i, sy_i);
            size_t idxo = xyz2idx(x, y, z, sx_o, sy_o);
            out[idxo] = in[idxi];
        }
    }
}