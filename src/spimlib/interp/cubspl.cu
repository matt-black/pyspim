#include <cuda_fp16.h>

inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
template<typename T>
inline __host__ __device__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
{
    const T one_frac = 1.0f - fraction;
    const T squared  = fraction * fraction;
    const T one_sqd  = one_frac * one_frac;

    w0 = 1.0f / 6.0f * one_sqd * one_frac;
    w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
    w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
    w3 = 1.0f / 6.0f * squared * fraction;
}
inline __host__ __device__ size_t xyz2idx(int x, int y, int z, 
                                            size_t width, size_t height)
{
    return x + ((z * height) + y) * width;
}
inline __host__ __device__ float lerp(float v0, float v1, float t)
{
    return fma(t, v1, fma(-t, v0, v0));
}
template<typename T>
__host__ __device__ float lerp3(const T* A,
                                int x_d, int y_d, int z_d,
                                float dx, float dy, float dz,
                                size_t width, size_t height)
{
    float data[2][2][2];
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
//comparison functions
inline __host__ __device__ bool geq0(const int3 a)
{
    return a.x >= 0 && a.y >= 0 && a.z >= 0;
}
inline __host__ __device__ bool operator<(const int3& a, const int3& b)
{
    return a.x < b.x && a.y < b.y && a.z < b.z;
}
//float3 functions
inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
inline __host__ __device__ float3 operator+(const float3& a, const float& b)
{
    return make_float3(a.x+b, a.y+b, a.z+b);
}
inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
inline __host__ __device__ float3 operator*(float a, const float3 b)
{
    return make_float3(a*b.x, a*b.y, a*b.z);
}
inline __host__ __device__ float3 operator/(const float3& a, const float3& b)
{
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}
inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
inline __host__ __device__ float3 operator-(const float3& a, const int3& b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
inline __host__ __device__ float3 operator-(float a, float3 b)
{
    return make_float3(a-b.x, a-b.y, a-b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x-b, a.y-b, a.z-b);
}
inline __host__ __device__ float3 floor(const float3 a)
{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}
inline __host__ __device__ int3 float3_rd(float3 a)
{
    return make_int3(__float2int_rd(a.x), __float2int_rd(a.y), __float2int_rd(a.z));
}


template<typename T>
__global__ void affineTransformCubSpl(float* out, T* in, float* M_aff,
                                      size_t sz_o, size_t sy_o, size_t sx_o,
                                      size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        const float3 coord = make_float3(
            dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])),
            dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])),
            dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]))
        );
        const float3 index = floor(coord);
        const float3 fract = coord - index;
        //calculate bspline weights and points
        float3 w0, w1, w2, w3;
        bspline_weights(fract, w0, w1, w2, w3);
        const float3 g0 = w0 + w1;
        const float3 g1 = w2 + w3;
        const float3 h0 = (w1 / g0) - 1 + index;
        const int3 h0i  = float3_rd(h0);
        const float3 h0f = h0 - h0i;
        const float3 h1 = (w3 / g1) + 1 + index;
        const int3 h1i  = float3_rd(h1);
        const float3 h1f = h1 - h1i;
        const int3 vol_dim = make_int3(sx_i-1, sy_i-1, sz_i-1);
        if (geq0(h0i) && geq0(h1i) && h0i < vol_dim && h1i < vol_dim) {
            float data000 = lerp3(in, h0i.x, h0i.y, h0i.z, h0f.x, h0f.y, h0f.z, sx_i, sy_i);
            float data100 = lerp3(in, h1i.x, h0i.y, h0i.z, h1f.x, h0f.y, h0f.z, sx_i, sy_i);
            data000 = g0.x * data000 + g1.x * data100;
            float data010 = lerp3(in, h0i.x, h1i.y, h0i.z, h0f.x, h1f.y, h0f.z, sx_i, sy_i);
            float data110 = lerp3(in, h1i.x, h1i.y, h0i.z, h1f.x, h1f.y, h0f.z, sx_i, sy_i);
            data010 = g0.x * data010 + g1.x * data110;
            data000 = g0.y * data000 + g1.y * data010;
            float data001 = lerp3(in, h0i.x, h0i.y, h1i.z, h0f.x, h0f.y, h1f.z, sx_i, sy_i);
            float data101 = lerp3(in, h1i.x, h0i.y, h1i.z, h1f.x, h0f.y, h1f.z, sx_i, sy_i);
            data001 = g0.x * data001 + g1.x * data101;
            float data011 = lerp3(in, h0i.x, h1i.y, h1i.z, h0f.x, h1f.y, h1f.z, sx_i, sy_i);
            float data111 = lerp3(in, h1i.x, h1i.y, h1i.z, h1f.x, h1f.y, h1f.z, sx_i, sy_i);
            data011 = g0.x * data011 + g1.x * data111;
            data001 = g0.y * data001 + g1.y * data011;
            // final interpolation
            size_t idx = xyz2idx(x, y, z, sx_o, sy_o);
            out[idx] = g0.z * data000 + g1.z * data001;
        }
    }
}


extern "C"
__global__ void affineTransformMaxBlend(unsigned short *out, unsigned short* in,
                                        float *M_aff, 
                                        size_t sz_o, size_t sy_o, size_t sx_o,
                                        size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        const float3 coord = make_float3(
            dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])),
            dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])),
            dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]))
        );
        const float3 index = floor(coord);
        const float3 fract = coord - index;
        //calculate bspline weights and points
        float3 w0, w1, w2, w3;
        bspline_weights(fract, w0, w1, w2, w3);
        const float3 g0 = w0 + w1;
        const float3 g1 = w2 + w3;
        const float3 h0 = (w1 / g0) - 1 + index;
        const int3 h0i  = float3_rd(h0);
        const float3 h0f = h0 - h0i;
        const float3 h1 = (w3 / g1) + 1 + index;
        const int3 h1i  = float3_rd(h1);
        const float3 h1f = h1 - h1i;
        const int3 vol_dim = make_int3(sx_i-1, sy_i-1, sz_i-1);
        if (geq0(h0i) && geq0(h1i) && h0i < vol_dim && h1i < vol_dim) {
            float data000 = lerp3(in, h0i.x, h0i.y, h0i.z, h0f.x, h0f.y, h0f.z, sx_i, sy_i);
            float data100 = lerp3(in, h1i.x, h0i.y, h0i.z, h1f.x, h0f.y, h0f.z, sx_i, sy_i);
            data000 = g0.x * data000 + g1.x * data100;
            float data010 = lerp3(in, h0i.x, h1i.y, h0i.z, h0f.x, h1f.y, h0f.z, sx_i, sy_i);
            float data110 = lerp3(in, h1i.x, h1i.y, h0i.z, h1f.x, h1f.y, h0f.z, sx_i, sy_i);
            data010 = g0.x * data010 + g1.x * data110;
            data000 = g0.y * data000 + g1.y * data010;
            float data001 = lerp3(in, h0i.x, h0i.y, h1i.z, h0f.x, h0f.y, h1f.z, sx_i, sy_i);
            float data101 = lerp3(in, h1i.x, h0i.y, h1i.z, h1f.x, h0f.y, h1f.z, sx_i, sy_i);
            data001 = g0.x * data001 + g1.x * data101;
            float data011 = lerp3(in, h0i.x, h1i.y, h1i.z, h0f.x, h1f.y, h1f.z, sx_i, sy_i);
            float data111 = lerp3(in, h1i.x, h1i.y, h1i.z, h1f.x, h1f.y, h1f.z, sx_i, sy_i);
            data011 = g0.x * data011 + g1.x * data111;
            data001 = g0.y * data001 + g1.y * data011;
            // final interpolation
            float ival = g0.z * data000 + g1.z * data001;
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
__global__ void affineTransformMeanBlend(half* *sum, 
                                         unsigned char* count,
                                         unsigned short* in,
                                         float *M_aff, 
                                         size_t sz_o, size_t sy_o, size_t sx_o,
                                         size_t sz_i, size_t sy_i, size_t sx_i)
{
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;
    if (x < sx_o && y < sy_o && z < sz_o) {
        float4 voxel = make_float4(x, y, z, 1.0f);
        const float3 coord = make_float3(
            dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])),
            dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])),
            dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11]))
        );
        const float3 index = floor(coord);
        const float3 fract = coord - index;
        //calculate bspline weights and points
        float3 w0, w1, w2, w3;
        bspline_weights(fract, w0, w1, w2, w3);
        const float3 g0 = w0 + w1;
        const float3 g1 = w2 + w3;
        const float3 h0 = (w1 / g0) - 1 + index;
        const int3 h0i  = float3_rd(h0);
        const float3 h0f = h0 - h0i;
        const float3 h1 = (w3 / g1) + 1 + index;
        const int3 h1i  = float3_rd(h1);
        const float3 h1f = h1 - h1i;
        const int3 vol_dim = make_int3(sx_i-1, sy_i-1, sz_i-1);
        if (geq0(h0i) && geq0(h1i) && h0i < vol_dim && h1i < vol_dim) {
            float data000 = lerp3(in, h0i.x, h0i.y, h0i.z, h0f.x, h0f.y, h0f.z, sx_i, sy_i);
            float data100 = lerp3(in, h1i.x, h0i.y, h0i.z, h1f.x, h0f.y, h0f.z, sx_i, sy_i);
            data000 = g0.x * data000 + g1.x * data100;
            float data010 = lerp3(in, h0i.x, h1i.y, h0i.z, h0f.x, h1f.y, h0f.z, sx_i, sy_i);
            float data110 = lerp3(in, h1i.x, h1i.y, h0i.z, h1f.x, h1f.y, h0f.z, sx_i, sy_i);
            data010 = g0.x * data010 + g1.x * data110;
            data000 = g0.y * data000 + g1.y * data010;
            float data001 = lerp3(in, h0i.x, h0i.y, h1i.z, h0f.x, h0f.y, h1f.z, sx_i, sy_i);
            float data101 = lerp3(in, h1i.x, h0i.y, h1i.z, h1f.x, h0f.y, h1f.z, sx_i, sy_i);
            data001 = g0.x * data001 + g1.x * data101;
            float data011 = lerp3(in, h0i.x, h1i.y, h1i.z, h0f.x, h1f.y, h1f.z, sx_i, sy_i);
            float data111 = lerp3(in, h1i.x, h1i.y, h1i.z, h1f.x, h1f.y, h1f.z, sx_i, sy_i);
            data011 = g0.x * data011 + g1.x * data111;
            data001 = g0.y * data001 + g1.y * data011;
            // final interpolation
            half ival = __float2half(g0.z * data000 + g1.z * data001);
            size_t idx = xyz2idx(x, y, z, sx_o, sy_o);
            count[idx] += 1;
            sum[idx] += ival;
        }
    }
}
