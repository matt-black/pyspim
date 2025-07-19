#include <cub/cub.cuh>

#define BLOCK_SZEX {block_size_x:d}
#define BLOCK_SZEY {block_size_y:d}
#define BLOCK_SZEZ {block_size_z:d}


inline __host__ __device__ float dot(float4 a, float4 b)
{{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}}

template<typename T>
inline __host__ __device__ void bspline_weights(T fraction, 
                                                T& w0, T& w1, T& w2, T& w3)
{{
    const T one_frac = 1.0f - fraction;
    const T squared  = fraction * fraction;
    const T one_sqd  = one_frac * one_frac;

    w0 = 1.0f / 6.0f * one_sqd * one_frac;
    w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
    w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
    w3 = 1.0f / 6.0f * squared * fraction;
}}

inline __host__ __device__ size_t xyz2idx(int x, int y, int z, 
                                            size_t width, size_t height)
{{
    return x + ((z * height) + y) * width;
}}

template<typename T>
inline __host__ __device__ T lerp(T v0, T v1, T t)
{{
    return fma(t, v1, fma(-t, v0, v0));
}}

template<typename T, typename U>
__host__ __device__ float lerp3(T* A, int x_d, int y_d, int z_d,
                                U dx, U dy, U dz,
                                size_t width, size_t height)
{{
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
}}

//comparison functions
inline __host__ __device__ bool geq0(const int3 a)
{{
    return a.x >= 0 && a.y >= 0 && a.z >= 0;
}}

inline __host__ __device__ bool operator<(const int3& a, const int3& b)
{{
    return a.x < b.x && a.y < b.y && a.z < b.z;
}}

//float3 functions
inline __host__ __device__ float3 operator+(const float3& a, const float3& b)
{{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}}

__host__ __device__ float3 operator+=(float3& a, const float3& b)
{{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}}

inline __host__ __device__ float3 operator+(const float3& a, const int b)
{{
    return make_float3(a.x+b, a.y+b, a.z+b);
}}

inline __host__ __device__ float3 operator+(const float3& a, const int3& b)
{{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}}

inline __host__ __device__ float3 operator*(const float3& a, const float3& b)
{{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}}

inline __host__ __device__ float3 operator*(float a, const float3 b)
{{
    return make_float3(a*b.x, a*b.y, a*b.z);
}}

inline __host__ __device__ float3 operator/(const float3& a, const float3& b)
{{
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b)
{{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}}

inline __host__ __device__ float3 operator-(const float3& a, const int3& b)
{{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}}

inline __host__ __device__ float3 operator-(float a, float3 b)
{{
    return make_float3(a-b.x, a-b.y, a-b.z);
}}

inline __host__ __device__ float3 operator-(float3 a, float b)
{{
    return make_float3(a.x-b, a.y-b, a.z-b);
}}

inline __host__ __device__ float3 floor(const float3 a)
{{
    return make_float3(floor(a.x), floor(a.y), floor(a.z));
}}

inline __host__ __device__ int3 float3_rd(float3 a)
{{
    return make_int3(__float2int_rd(a.x), 
                     __float2int_rd(a.y), 
                     __float2int_rd(a.z));
}}


extern "C"
__global__ void {function_name:s}(float* prods, float* M_aff,
                                  unsigned short* reference, 
                                  unsigned short* moving,
                                  float mu_ref, float mu_mov,
                                  size_t sz_r, size_t sy_r, size_t sx_r,
                                  size_t sz_m, size_t sy_m, size_t sx_m)
{{
    // NOTE: input data should be in ZRC format, but CUDA grids are XYZ
    const size_t z = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t x = blockDim.z * blockIdx.z + threadIdx.z;

    float3 vals = make_float3(0.0f, 0.0f, 0.0f);
    if (z < sz_m && y < sy_m && x < sx_m) {{
        float4 voxel = make_float4(x, y, z, 1.0f);
        const float3 coord = make_float3(
            dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])),
            dot(voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])),
            dot(voxel, make_float4(M_aff[8], M_aff[9], M_aff[10],M_aff[11]))
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
        const int3 vol_dim = make_int3(sx_r-1, sy_r-1, sz_r-1);
        if (geq0(h0i) && geq0(h1i) && h0i < vol_dim && h1i < vol_dim) {{
            // do bicubic spline interpolation on reference
            float data000 = lerp3(
                reference, h0i.x, h0i.y, h0i.z, h0f.x, h0f.y, h0f.z, sx_r, sy_r
            );
            float data100 = lerp3(
                reference, h1i.x, h0i.y, h0i.z, h1f.x, h0f.y, h0f.z, sx_r, sy_r
            );
            data000 = g0.x * data000 + g1.x * data100;
            float data010 = lerp3(
                reference, h0i.x, h1i.y, h0i.z, h0f.x, h1f.y, h0f.z, sx_r, sy_r
            );
            float data110 = lerp3(
                reference, h1i.x, h1i.y, h0i.z, h1f.x, h1f.y, h0f.z, sx_r, sy_r
            );
            data010 = g0.x * data010 + g1.x * data110;
            data000 = g0.y * data000 + g1.y * data010;
            float data001 = lerp3(
                reference, h0i.x, h0i.y, h1i.z, h0f.x, h0f.y, h1f.z, sx_r, sy_r
            );
            float data101 = lerp3(
                reference, h1i.x, h0i.y, h1i.z, h1f.x, h0f.y, h1f.z, sx_r, sy_r
            );
            data001 = g0.x * data001 + g1.x * data101;
            float data011 = lerp3(
                reference, h0i.x, h1i.y, h1i.z, h0f.x, h1f.y, h1f.z, sx_r, sy_r
            );
            float data111 = lerp3(
                reference, h1i.x, h1i.y, h1i.z, h1f.x, h1f.y, h1f.z, sx_r, sy_r
            );
            data011 = g0.x * data011 + g1.x * data111;
            data001 = g0.y * data001 + g1.y * data011;
            float rval = g0.z * data000 + g1.z * data001;
            // grab equivalent value from moving
            float mval = (float)moving[xyz2idx(x, y, z, sx_m, sy_m)];
            vals = make_float3({expr1:s}, {expr2:s}, {expr3:s});
        }} // else {{ do nothing; out of range }}
    }}

    typedef cub::BlockReduce<float3, BLOCK_SZEX, cub::BLOCK_REDUCE_RAKING, BLOCK_SZEY, BLOCK_SZEZ> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float3 aggregate = BlockReduce(temp_storage).Sum(vals);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{ 
        atomicAdd(&prods[0], vals.x);
        atomicAdd(&prods[1], vals.y);
        atomicAdd(&prods[2], vals.z);
    }}
}}