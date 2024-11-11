inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z + b.z + a.w * b.w;
}

inline __host__ __device__ size_t xyz2idx(int x, int y, int z, 
                                            size_t width, size_t height)
{
    return x + ((z * height) + y) * width;
}

template<typename T>
inline __host__ __device__ T lerp(T v0, T v1, T t)
{
    return fma(t, v1, fma(-t, v0, v0));
}

__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return float3{a.x+b.x, a.y+b.y, a.z+b.z};
}

__host__ __device__ float3 operator+=(float3& a, const float3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return float3{a.x-b.x, a.y-b.y, a.z-b.z};
}

template<typename T, typename U>
__host__ __device__ U lerp3(const T* A,
                            int x_rd, int y_rd, int z_rd,
                            U dx, U dy, U dz,
                            size_t width, size_t height)
{
    //collect data from array
    U data[2][2][2];
    data[0][0][0] = A[xyz2idx(x_rd,   y_rd,   z_rd,   width, height)];
    data[0][0][1] = A[xyz2idx(x_rd,   y_rd,   z_rd+1, width, height)];
    data[0][1][1] = A[xyz2idx(x_rd,   y_rd+1, z_rd+1, width, height)];
    data[0][1][0] = A[xyz2idx(x_rd,   y_rd+1, z_rd,   width, height)];
    data[1][1][0] = A[xyz2idx(x_rd+1, y_rd+1, z_rd,   width, height)];
    data[1][1][1] = A[xyz2idx(x_rd+1, y_rd+1, z_rd+1, width, height)];
    data[1][0][1] = A[xyz2idx(x_rd+1, y_rd,   z_rd+1, width, height)];
    data[1][0][0] = A[xyz2idx(x_rd+1, y_rd,   z_rd,   width, height)];
    // do trilinear interpolation on the collected data
    return lerp(lerp(lerp(data[0][0][0], data[0][0][1], dz),
                     lerp(data[0][1][0], data[0][1][1], dz), dy),
                lerp(lerp(data[1][0][0], data[1][0][1], dz),
                     lerp(data[1][1][0], data[1][1][1], dz), dy),
                dx);
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
                float x_r = dot(voxel, make_float4(
                                M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
                int x_rd  = __float2int_rd(x_r);
                U dx  = x_r - (U)x_rd;
                float y_r = dot(voxel, 
                                make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
                int y_rd  = __float2int_rd(y_r);
                U dy  = y_r - (U)y_rd;
                float z_r = dot(voxel, 
                                make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
                int z_rd  = __float2int_rd(z_r);
                U dz  = z_r - (U)z_rd;
                if (x_rd >= 0 && x_rd < sx_r-1 && 
                    y_rd >= 0 && y_rd < sy_r-1 && 
                    z_rd >= 0 && z_rd < sz_r-1) {
                    //collect data from array
                    U rval = lerp3(
                        reference, x_rd, y_rd, z_rd, dx, dy, dz, sx_r, sy_r
                    );
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += rval * mval;
                    vals[1] += rval * rval;
                    vals[2] += mval * mval;
                } // else { do nothing; out of range }
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
    const size_t z0 = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t z_stride = blockDim.x * gridDim.x;
    const size_t y0 = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t y_stride = blockDim.y * gridDim.y;
    const size_t x0 = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t x_stride = blockDim.z * gridDim.z;

    U vals[3] = {};
    for (int z = z0; z < sz_m; z += z_stride) {
        for (int y = y0; y < sy_m; y += y_stride) {
            for (int x = x0; x < sx_m; x += x_stride) {
                float4 voxel = make_float4(x, y, z, 1.0f);
                float x_r = dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
                int x_rd  = __float2int_rd(x_r);
                U dx  = x_r - (U)x_rd;
                float y_r = dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
                int y_rd  = __float2int_rd(y_r);
                U dy  = y_r - (U)y_rd;
                float z_r = dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
                int z_rd  = __float2int_rd(z_r);
                U dz  = z_r - (U)z_rd;
                if (x_rd >= 0 && x_rd < sx_r-1 && 
                    y_rd >= 0 && y_rd < sy_r-1 && 
                    z_rd >= 0 && z_rd < sz_r-1) {
                    U rval = lerp3(
                        reference, x_rd, y_rd, z_rd, dx, dy, dz, sx_r, sy_r
                    );
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += rval * mval;
                    vals[1] += (rval - mu_ref) * (rval - mu_ref);
                    vals[2] += mval * mval;
                } // else { do nothing; out of range }
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

    U vals[3] = {};
    for (int z = z0; z < sz_m; z += z_stride) {
        for (int y = y0; y < sy_m; y += y_stride) {
            for (int x = x0; x < sx_m; x += x_stride) {
                float4 voxel = make_float4(x, y, z, 1.0f);
                float x_r = dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
                int x_rd  = __float2int_rd(x_r);
                U dx  = x_r - (U)x_rd;
                float y_r = dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
                int y_rd  = __float2int_rd(y_r);
                U dy  = y_r - (U)y_rd;
                float z_r = dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
                int z_rd  = __float2int_rd(z_r);
                U dz  = z_r - (U)z_rd;
                if (x_rd >= 0 && x_rd < sx_r-1 && 
                    y_rd >= 0 && y_rd < sy_r-1 && 
                    z_rd >= 0 && z_rd < sz_r-1) {
                    U rval = lerp3(
                        reference, x_rd, y_rd, z_rd, dx, dy, dz, sx_r, sy_r
                    );
                    U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                    vals[0] += (rval-mu_ref)*(mval-mu_mov);
                    vals[1] += (rval-mu_ref)*(rval-mu_ref);
                    vals[2] += (mval-mu_mov)*(mval-mu_mov);
                } // else { do nothing; out of range }
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