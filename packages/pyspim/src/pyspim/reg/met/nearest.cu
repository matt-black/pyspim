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

template<typename T, typename U>
__global__ void normInnerProduct(U* prods, float* M_aff,
                                 T* reference, T* moving,
                                 size_t sz_r, size_t sy_r, size_t sx_r,
                                 size_t sz_m, size_t sy_m, size_t sx_m)
{
    // NOTE: input data should be in ZRC format, but CUDA grids are XYZ
    const size_t z0 = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t z_stride = blockDim.x * gridDim.x;
    const size_t y0 = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t y_stride = blockDim.y * gridDim.y;
    const size_t x0 = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t x_stride = blockDim.z * gridDim.z;

    U vals[3] = {};  //initialize to zero
    for (int z = z0; z < sz_m; z += z_stride) {
        for (int y = y0; y < sy_m; y += y_stride) {
            for (int x = x0; x < sx_m; x += x_stride) {
                float4 voxel = make_float4(x, y, z, 1.0f);
                float x_t = dot(
                    voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])
                );
                int x_i = nearestNeighbor(x_t);
                float y_t = dot(
                    voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
                );
                int y_i = nearestNeighbor(y_t);
                float z_t = dot(
                    voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
                );
                int z_i = nearestNeighbor(z_t);
                if (x_i >= 0 && x_i < sx_r && y_i >= 0 && y_i < sy_r &&
                    z_i >= 0 && z_i < sz_r) {
                    U rval = (U)reference[xyz2idx(x_i, y_i, z_i, sx_r, sy_r)];
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += rval * mval;
                    vals[1] += rval * rval;
                    vals[2] += mval * mval;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicAdd(&prods[0], vals[0]);
        atomicAdd(&prods[1], vals[1]);
        atomicAdd(&prods[2], vals[2]);
    }
}

template<typename T, typename U>
__global__ void correlationRatio(U* prods, float* M_aff,
                                 T* reference, T* moving, U mu_ref,
                                 size_t sz_r, size_t sy_r, size_t sx_r,
                                 size_t sz_m, size_t sy_m, size_t sx_m)
{
    // NOTE: input data should be in ZRC format, but CUDA grids are XYZ
    const size_t z0 = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t z_stride = blockDim.x * gridDim.x;
    const size_t y0 = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t y_stride = blockDim.y * gridDim.y;
    const size_t x0 = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t x_stride = blockDim.z * gridDim.z;

    U vals[3] = {};  //initialize to zero
    for (int z = z0; z < sz_m; z += z_stride) {
        for (int y = y0; y < sy_m; y += y_stride) {
            for (int x = x0; x < sx_m; x += x_stride) {
                float4 voxel = make_float4(x, y, z, 1.0f);
                float x_t = dot(
                    voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])
                );
                int x_i = nearestNeighbor(x_t);
                float y_t = dot(
                    voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
                );
                int y_i = nearestNeighbor(y_t);
                float z_t = dot(
                    voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
                );
                int z_i = nearestNeighbor(z_t);
                if (x_i >= 0 && x_i < sx_r && y_i >= 0 && y_i < sy_r &&
                    z_i >= 0 && z_i < sz_r) {
                    U rval = (U)reference[xyz2idx(x_i, y_i, z_i, sx_r, sy_r)];
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += rval * mval;
                    vals[1] += (rval - mu_ref) * (rval - mu_ref);
                    vals[2] += mval * mval;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicAdd(&prods[0], vals[0]);
        atomicAdd(&prods[1], vals[1]);
        atomicAdd(&prods[2], vals[2]);
    }
}

template<typename T, typename U>
__global__ void normCrossCorrelation(U* prods, float* M_aff,
                                     T* reference, T* moving, 
                                     U mu_ref, U mu_mov,
                                     size_t sz_r, size_t sy_r, size_t sx_r,
                                     size_t sz_m, size_t sy_m, size_t sx_m)
{
    // NOTE: input data should be in ZRC format, but CUDA grids are XYZ
    const size_t z0 = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t z_stride = blockDim.x * gridDim.x;
    const size_t y0 = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t y_stride = blockDim.y * gridDim.y;
    const size_t x0 = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t x_stride = blockDim.z * gridDim.z;

    U vals[3] = {};  //initialize to zero
    for (int z = z0; z < sz_m; z += z_stride) {
        for (int y = y0; y < sy_m; y += y_stride) {
            for (int x = x0; x < sx_m; x += x_stride) {
                float4 voxel = make_float4(x, y, z, 1.0f);
                float x_t = dot(
                    voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])
                );
                int x_i = nearestNeighbor(x_t);
                float y_t = dot(
                    voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
                );
                int y_i = nearestNeighbor(y_t);
                float z_t = dot(
                    voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
                );
                int z_i = nearestNeighbor(z_t);
                if (x_i >= 0 && x_i < sx_r && y_i >= 0 && y_i < sy_r &&
                    z_i >= 0 && z_i < sz_r) {
                    U rval = (U)reference[xyz2idx(x_i, y_i, z_i, sx_r, sy_r)];
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += (rval - mu_ref) * (mval - mu_mov);
                    vals[1] += (rval - mu_ref) * (rval - mu_ref);
                    vals[2] += (mval - mu_mov) * (mval - mu_mov);
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicAdd(&prods[0], vals[0]);
        atomicAdd(&prods[1], vals[1]);
        atomicAdd(&prods[2], vals[2]);
    }
}