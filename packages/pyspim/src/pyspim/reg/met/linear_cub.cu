#include <cub/cub.cuh>

#define BLOCK_SZEX {block_size_x:d}
#define BLOCK_SZEY {block_size_y:d}
#define BLOCK_SZEZ {block_size_z:d}


inline __host__ __device__ float dot(float4 a, float4 b)
{{
    return a.x * b.x + a.y * b.y + a.z + b.z + a.w * b.w;
}}


inline __host__ __device__ size_t xyz2idx(size_t x, size_t y, size_t z,
                                          size_t sze_x, size_t sze_y)
{{
    return x + y * sze_x + z * sze_x * sze_y;
}}


inline __host__ __device__ float lerp(float v0, float v1, float t)
{{
    return fma(t, v1, fma(-t, v0, v0));
}}


__host__ __device__ float lerp3(const unsigned short* A,
                                int x_rd, int y_rd, int z_rd,
                                float dx, float dy, float dz,
                                size_t width, size_t height)
{{
    //collect data from array
    float data[2][2][2];
    data[0][0][0] = (float)A[xyz2idx(x_rd,   y_rd,   z_rd,   width, height)];
    data[0][0][1] = (float)A[xyz2idx(x_rd,   y_rd,   z_rd+1, width, height)];
    data[0][1][1] = (float)A[xyz2idx(x_rd,   y_rd+1, z_rd+1, width, height)];
    data[0][1][0] = (float)A[xyz2idx(x_rd,   y_rd+1, z_rd,   width, height)];
    data[1][1][0] = (float)A[xyz2idx(x_rd+1, y_rd+1, z_rd,   width, height)];
    data[1][1][1] = (float)A[xyz2idx(x_rd+1, y_rd+1, z_rd+1, width, height)];
    data[1][0][1] = (float)A[xyz2idx(x_rd+1, y_rd,   z_rd+1, width, height)];
    data[1][0][0] = (float)A[xyz2idx(x_rd+1, y_rd,   z_rd,   width, height)];
    // do trilinear interpolation on the collected data
    return lerp(lerp(lerp(data[0][0][0], data[0][0][1], dz),
                     lerp(data[0][1][0], data[0][1][1], dz), dy),
                lerp(lerp(data[1][0][0], data[1][0][1], dz),
                     lerp(data[1][1][0], data[1][1][1], dz), dy),
                dx);
}}

__host__ __device__ float3 operator+(const float3& a, const float3& b)
{{
    return float3{{a.x+b.x, a.y+b.y, a.z+b.z}};
}}


__host__ __device__ float3 operator+=(float3& a, const float3& b)
{{
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}}


__host__ __device__ float3 operator-(const float3& a, const float3& b)
{{
    return float3{{a.x-b.x, a.y-b.y, a.z-b.z}};
}}


extern "C"
__global__ void {function_name:s}(float* vals, float* M_aff, 
                                  unsigned short* moving, 
                                  unsigned short* reference,
                                  float mu_ref, float mu_mov,
                                  size_t sx_m, size_t sy_m, size_t sz_m,
                                  size_t sx_r, size_t sy_r, size_t sz_r)
{{
    /* compute the inner product between the reference and moving images
    where the moving image is transformed by some affine transform M_aff^{{-1}}

    outputs: vals[0] holds r \dot m, vals[1] holds r \dot r
    */
    const size_t x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t z = blockDim.z * blockIdx.z + threadIdx.z;

    float3 prods = make_float3(0.0f, 0.0f, 0.0f);
    if (x < sx_m && y < sy_m && z < sz_m) {{
        //compute the coordinate in the reference image
        float4 voxel = make_float4(x, y, z, 1.0f);
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
        // range-check that the transformed coordinate is valid
        if (x_rd >= 0 && x_rd < sx_r-1 && 
            y_rd >= 0 && y_rd < sy_r-1 && 
            z_rd >= 0 && z_rd < sz_r-1) {{
            float rval = lerp3(
                reference, x_rd, y_rd, z_rd, dx, dy, dz, sx_r, sy_r
            );
            //fetch value for moving image at this index
            int idx = xyz2idx(x, y, z, sx_m, sy_m);
            float mval = (float)moving[idx];
            prods = make_float3(
                {expr1:s}, {expr2:s}, {expr3:s}
            );
        }} // else {{ do nothing; }}
    }}

    // handle the reduction
    typedef cub::BlockReduce<float3, BLOCK_SZEX, cub::BLOCK_REDUCE_RAKING, BLOCK_SZEY, BLOCK_SZEZ> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float3 aggregate = BlockReduce(temp_storage).Sum(prods);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {{ 
        atomicAdd(&vals[0], aggregate.x);
        atomicAdd(&vals[1], aggregate.y);
        atomicAdd(&vals[2], aggregate.z);
    }}
}}