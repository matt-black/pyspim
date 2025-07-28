/*
    CUDA kernel (and supporting functions) for computing image registration metrics

    This file requires string formatting, expecting the following variables to be declared:
        1. interp_type (str): one of CUBSPL, LINEAR, NEAREST
            This controls the #ifdef/#endif statement inside the kernel that determines how the interpolation is done. 
        2. function_name (str): the name to be given to the kernel function
            (should be one of normInnerProduct, correlationRatio, normCrossCorrelation)
        3. expr1 (str): how values from the moving and reference are combined
        4. expr2 (str): how values from the reference are squared
        5. expr3 (str): how values from the moving are squared

    The output metric will be computed like (expr1) / (expr2 * expr3).

    To use this, read this file in as text in Python, then format the string like:

    ```
    with open("/path/to/this/file") as f:
        txt = f.read()
    txt_to_compile = txt.format(
        function_name=function_name,
        interp_type=interp_type,
        expr1=expr1,
        expr2=expr2,
        expr3=expr3
    )
    ```

    This text can then be compiled using cupy.RawModule
*/
#define {interp_type:s}  // one of CUBSPL, LINEAR, NEAREST


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

#ifdef NEAREST
inline __host__ __device__ int nearestNeighbor(float coord)
{{
    if (fmodf(coord, 1.0f) < 0.5f) {{
        return __float2int_rd(coord);
    }} else {{
        return __float2int_rd(coord)+1;
    }}
}}
#endif

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

inline __host__ __device__ float3 operator+(const float3& a, const float& b)
{{
    return make_float3(a.x+b, a.y+b, a.z+b);
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

template<typename T, typename U>
__global__ void {function_name:s}(
    U* prods, float* M_aff,
    T* reference, T* moving,
    float* mu_ref, float* mu_mov,
    size_t sz_r, size_t sy_r, size_t sx_r,
    size_t sz_m, size_t sy_m, size_t sx_m,
    int gpu_id, int num_gpu
)
{{
    const size_t z0 = blockDim.z * blockIdx.z + threadIdx.z;
    const size_t z_stride = blockDim.z * gridDim.z;
    const size_t y0 = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t y_stride = blockDim.y * gridDim.y;
    const size_t x0 = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t x_stride = blockDim.x * gridDim.x;

    const int3 vol_dim = make_int3(sx_r-1, sy_r-1, sz_r-1);
    // determine chunk for this gpu
    int z_per_gpu = (sz_m + num_gpu - 1) / num_gpu;
    int z_start = gpu_id * z_per_gpu;
    int z_end = min((gpu_id + 1) * z_per_gpu, int(sz_m));

    __shared__ U vals[1024][3];
    // determine which thread we're in, initialize values to 0
    int thread_id = xyz2idx(threadIdx.x, threadIdx.y, threadIdx.z,
                            blockDim.x, blockDim.y);
    vals[thread_id][0] = 0;
    vals[thread_id][1] = 0;
    vals[thread_id][2] = 0;

    for (int z = z0; z < z_end; z += z_stride) {{
        if (z >= z_start) {{
            for (int y = y0; y < sy_m; y += y_stride) {{
                for (int x = x0; x < sx_m; x += x_stride) {{
                    // get current voxel and coordinate in transformed coordinate system
                    float4 voxel = make_float4(x, y, z, 1.0f);
                    #ifdef CUBSPL
                        const float3 coord = make_float3(
                            dot(voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])),
                            dot(voxel, make_float4(M_aff[4],  M_aff[5],  M_aff[6],  M_aff[7])),
                            dot(voxel, make_float4(M_aff[8],  M_aff[9],  M_aff[10], M_aff[11]))
                        );
                        // determine the index of nearest point (by floor) and the weight
                        const float3 index = floor(coord);
                        const float3 fract = coord - index;
                        //calculate bspline weights and points
                        float3 w0, w1, w2, w3;
                        bspline_weights(fract, w0, w1, w2, w3);
                        const float3 g0 = w0 + w1;
                        const float3 g1 = w2 + w3;
                        const float3 h0 = (w1 / g0) - 1.0f + index;
                        const int3 h0i  = float3_rd(h0);
                        const float3 h0f = h0 - h0i;
                        const float3 h1 = (w3 / g1) + 1.0f + index;
                        const int3 h1i  = float3_rd(h1);
                        const float3 h1f = h1 - h1i;

                        if (geq0(h0i) && geq0(h1i) && h0i < vol_dim && h1i < vol_dim) {{
                            // do bicubic spline interpolation on reference
                            U data000 = lerp3(reference, h0i.x, h0i.y, h0i.z, 
                                          h0f.x, h0f.y, h0f.z, sx_r, sy_r);
                            U data100 = lerp3(reference, h1i.x, h0i.y, h0i.z, 
                                            h1f.x, h0f.y, h0f.z, sx_r, sy_r);
                            data000 = g0.x * data000 + g1.x * data100;
                            U data010 = lerp3(reference, h0i.x, h1i.y, h0i.z, 
                                            h0f.x, h1f.y, h0f.z, sx_r, sy_r);
                            U data110 = lerp3(reference, h1i.x, h1i.y, h0i.z, 
                                            h1f.x, h1f.y, h0f.z, sx_r, sy_r);
                            data010 = g0.x * data010 + g1.x * data110;
                            data000 = g0.y * data000 + g1.y * data010;
                            U data001 = lerp3(reference, h0i.x, h0i.y, h1i.z, 
                                            h0f.x, h0f.y, h1f.z, sx_r, sy_r);
                            U data101 = lerp3(reference, h1i.x, h0i.y, h1i.z, 
                                            h1f.x, h0f.y, h1f.z, sx_r, sy_r);
                            data001 = g0.x * data001 + g1.x * data101;
                            U data011 = lerp3(reference, h0i.x, h1i.y, h1i.z, 
                                            h0f.x, h1f.y, h1f.z, sx_r, sy_r);
                            U data111 = lerp3(reference, h1i.x, h1i.y, h1i.z, 
                                            h1f.x, h1f.y, h1f.z, sx_r, sy_r);
                            data011 = g0.x * data011 + g1.x * data111;
                            data001 = g0.y * data001 + g1.y * data011;
                            U rval = g0.z * data000 + g1.z * data001;
                            // grab equivalent value from moving
                            U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                            if (!isfinite(rval))  
                                printf("NaN/Inf at %d %d %d\n", x, y, z);

                            if (abs(mval) > 0.00001) {{
                                vals[thread_id][0] += {expr1:s}; // rval * mval;
                                vals[thread_id][1] += {expr2:s}; //(rval - mu_ref) * (rval - mu_ref);
                                vals[thread_id][2] += {expr3:s}; // mval * mval;
                            }}
                        }}
                    #endif
                    #ifdef LINEAR
                        float x_r = dot(
                            voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])
                        );
                        int x_rd = __float2int_rd(x_r);
                        float dx = x_r - (float)x_rd;
                        float y_r = dot(
                            voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
                        );
                        int y_rd = __float2int_rd(y_r);
                        float dy = y_r - (float)y_rd;
                        float z_r = dot(
                            voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
                        );
                        int z_rd = __float2int_rd(z_r);
                        float dz = z_r - (float)z_rd;
                        //range-check that transformed coordinate is valid
                        if (x_rd >= 0 && x_rd < sx_r-1 &&
                            y_rd >= 0 && y_rd < sy_r-1 &&
                            z_rd >= 0 && z_rd < sz_r-1) {{
                            U rval = lerp3(
                                reference, x_rd, y_rd, z_rd, dx, dy, dz, sx_r, sy_r
                            );
                            // fetch value for moving image at this index
                            int idx = xyz2idx(x, y, z, sx_m, sy_m);
                            U mval = (U)moving[idx];
                            if (abs(mval) > 0.00001) {{
                                vals[thread_id][0] += {expr1:s};
                                vals[thread_id][1] += {expr2:s};
                                vals[thread_id][2] += {expr3:s};
                            }}
                        }}
                    #endif
                    #ifdef NEAREST
                        float x_t = dot(
                            voxel, make_float4(M_aff[0], M_aff[1], M_aff[2], M_aff[3])
                        );
                        float y_t = dot(
                            voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
                        );
                        float z_t = dot(
                            voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
                        );
                        int x_i = nearestNeighbor(x_t);
                        int y_i = nearestNeighbor(y_t);
                        int z_i = nearestNeighbor(z_t);
                        if (x_i >= 0 && x_i < sx_r && y_i >= 0 && y_i < sy_r &&
                            z_i >= 0 && z_i < sz_r) {{
                            U rval = (U)reference[xyz2idx(x_i, y_i, z_i, sx_r, sy_r)];
                            U mval = (U)moving[xyz2idx(x, y, z, sx_m, sy_m)];
                            vals[thread_id][0] += {expr1:s}; //rval * mval;
                            vals[thread_id][1] += {expr2:s}; //rval * rval;
                            vals[thread_id][2] += {expr3:s}; //mval * mval;
                        }}
                    #endif
                }}
            }}
        }}
    }}

    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{
        //reduce all thre threads in the block to a single value, stored in the
        // 0 index of vals.
        for (int i = 1; i < blockDim.x * blockDim.y * blockDim.z; i++) {{
            vals[0][0] += vals[i][0];
            vals[0][1] += vals[i][1];
            vals[0][2] += vals[i][2];
        }}
        atomicAdd(&prods[0], vals[0][0]);
        atomicAdd(&prods[1], vals[0][1]);
        atomicAdd(&prods[2], vals[0][2]);
    }}
}}
