#include <cuda_fp16.h>

inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<typename T>
inline __host__ __device__ T lerp(T v0, T v1, T t)
{
    return fma(t, v1, fma(-t, v0, v0));
}

inline __host__ __device__ size_t xyz2idx(int x, int y, int z, 
                                          size_t width, size_t height)
{
    return x + ((z * height) + y) * width;
}

template<typename T, typename U>
__host__ __device__ U lerp3(const T* A,
                            int x_d, int y_d, int z_d,
                            U dx, U dy, U dz,
                            size_t width, size_t height)
{
    U data[2][2][2];
    data[0][0][0] = A[xyz2idx(x_d,   y_d,   z_d,   width, height)];
    data[0][0][1] = A[xyz2idx(x_d,   y_d,   z_d+1, width, height)];
    data[0][1][1] = A[xyz2idx(x_d,   y_d+1, z_d+1, width, height)];
    data[0][1][0] = A[xyz2idx(x_d,   y_d+1, z_d,   width, height)];
    data[1][0][0] = A[xyz2idx(x_d+1, y_d,   z_d,   width, height)];
    data[1][0][1] = A[xyz2idx(x_d+1, y_d,   z_d+1, width, height)];
    data[1][1][1] = A[xyz2idx(x_d+1, y_d+1, z_d+1, width, height)];
    data[1][1][0] = A[xyz2idx(x_d+1, y_d+1, z_d,   width, height)];
    return lerp(lerp(lerp(data[0][0][0], data[1][0][0], dx),
                     lerp(data[0][1][0], data[1][1][0], dx), dy),
                lerp(lerp(data[0][0][1], data[1][0][1], dx),
                     lerp(data[0][1][1], data[1][1][1], dx), dy),
                dz);
}


template<typename T>
__global__ void affineTransformLerp(float* out, T* in, float* M_aff,
                                    size_t sz_o, size_t sy_o, size_t sx_o,
                                    size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        float x_t = dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
        int x_td  = __float2int_rd(x_t);
        float dx  = x_t - x_td;
        float y_t = dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
        int y_td  = __float2int_rd(y_t);
        float dy  = y_t - y_td;
        float z_t = dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
        int z_td  = __float2int_rd(z_t);
        float dz  = z_t - z_td;
        if (x_td >= 0 && x_td < sx_i-1 && 
            y_td >= 0 && y_td < sy_i-1 && 
            z_td >= 0 && z_td < sz_i-1) {
            float ival = lerp3(in, x_td, y_td, z_td, dx, dy, dz, sx_i, sy_i);
            size_t idx = xyz2idx(x, y, z, sx_o, sy_o);
            out[idx] = ival;
        }
    }
}


extern "C"
__global__ void affineTransformMaxBlend(unsigned short *out, unsigned short* in,
                                        float* M_aff,
                                        size_t sz_o, size_t sy_o, size_t sx_o,
                                        size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        float x_t = dot(voxel, 
                        make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
        int x_td  = __float2int_rd(x_t);
        float dx  = x_t - x_td;
        float y_t = dot(voxel, 
                        make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
        int y_td  = __float2int_rd(y_t);
        float dy  = y_t - y_td;
        float z_t = dot(voxel, 
                        make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
        int z_td  = __float2int_rd(z_t);
        float dz  = z_t - z_td;
        if (x_td >= 0 && x_td < sx_i-1 && 
            y_td >= 0 && y_td < sy_i-1 && 
            z_td >= 0 && z_td < sz_i-1) {
            int ival = __float2int_rn(
                lerp3(in, x_td, y_td, z_td, dx, dy, dz, sx_i, sy_i)
            );
            if (ival > 0) {
                size_t idx = xyz2idx(x, y, z, sx_o, sy_o);
                if (ival > 65535) {  // largest possible uint16_t
                    out[idx] = 65535;
                } else {
                    out[idx] = max(out[idx], (unsigned short)ival);
                }
            }
        }
    }
}


extern "C"
__global__ void affineTransformMeanBlend(half* sum, 
                                         unsigned char* count,
                                         unsigned short* in,
                                         float* M_aff,
                                         size_t sz_o, size_t sy_o, size_t sx_o,
                                         size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        float x_t = dot(voxel, 
                        make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3]));
        int x_td  = __float2int_rd(x_t);
        float dx  = x_t - x_td;
        float y_t = dot(voxel, 
                        make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7]));
        int y_td  = __float2int_rd(y_t);
        float dy  = y_t - y_td;
        float z_t = dot(voxel, 
                        make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]));
        int z_td  = __float2int_rd(z_t);
        float dz  = z_t - z_td;
        if (x_td >= 0 && x_td < sx_i-1 && 
            y_td >= 0 && y_td < sy_i-1 && 
            z_td >= 0 && z_td < sz_i-1) {
            half ival = __float2half(
                lerp3(in, x_td, y_td, z_td, dx, dy, dz, sx_i, sy_i)
            );
            size_t idx = xyz2idx(x, y, z, sx_o, sy_o);
            count[idx] += 1;
            sum[idx] += ival;
        }
    }
}