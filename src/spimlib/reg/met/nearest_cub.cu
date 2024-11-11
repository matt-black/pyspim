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

inline __host__ __device__ int nearestNeighbor(float coord)
{{
    if (fmodf(coord, 1.0) < 0.5f) {{
        return __float2int_rd(coord);
    }} else {{
        return __float2int_rd(coord)+1;
    }}
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


extern "C"
__global__ void {function_name:s}(float* vals, float* M_aff,
                                  unsigned short* moving,
                                  unsigned short* reference,
                                  float mu_ref, float mu_mov,
                                  size_t sx_m, size_t sy_m, size_t sz_m,
                                  size_t sx_r, size_t sy_r, size_t sz_r)
{{
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
        int x_i = nearestNeighbor(x_r);
        float y_r = dot(
            voxel, make_float4(M_aff[4], M_aff[5], M_aff[6], M_aff[7])
        );
        int y_i = nearestNeighbor(y_r);
        float z_r = dot(
            voxel, make_float4(M_aff[8], M_aff[9], M_aff[10], M_aff[11])
        );
        int z_i = nearestNeighbor(z_r);
        // range-check that the transformed coordinate is valid
        if (x_i >= 0 && x_i < sx_r-1 &&  y_i >= 0 && y_i < sy_r-1 && 
            z_i >= 0 && z_i < sz_r-1) {{
            float rval = (float)reference[xyz2idx(x_i, y_i, z_i, sx_r, sy_r)];
            float mval = (float)moving[xyz2idx(x, y, z, sx_m, sy_m)];
            prods = make_float3({expr1:s}, {expr2:s}, {expr3:s});
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