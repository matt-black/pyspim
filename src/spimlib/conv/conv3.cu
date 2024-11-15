#define KERNEL_RADIUS 31

//pre-determine # of threads per block (given by blockdim, total = x *y)
#define Z_BLOCKDIM_X   {z_blockdim_x:d}
#define Z_BLOCKDIM_Y   {z_blockdim_y:d}
//# of pixels in x convolved by each thread
#define Z_RESULT_STEPS {z_result_steps:d}
//# of border pixels
//effective width is ROWS_BORDER_PIX * 2 + ROWS_BLOCKDIM_X
#define Z_BORDER_PIX   {z_border_pix:d}

#define R_BLOCKDIM_X   {r_blockdim_x:d}
#define R_BLOCKDIM_Y   {r_blockdim_y:d}
#define R_RESULT_STEPS {r_result_steps:d}
#define R_BORDER_PIX   {r_border_pix:d}

#define C_BLOCKDIM_X   {c_blockdim_x:d}
#define C_BLOCKDIM_Z   {c_blockdim_z:d}
#define C_RESULT_STEPS {c_result_steps:d}
#define C_BORDER_PIX   {c_border_pix:d}


template<typename T>
__global__ void convZKernel(float* out, T* in, float* kernel,
                            const int sze_z, const int sze_r, const int sze_c)
{{
    __shared__ T shr_data[Z_BLOCKDIM_Y][(Z_RESULT_STEPS+2*Z_BORDER_PIX)
                                         *Z_BLOCKDIM_X];
    const int z0 = (blockIdx.x * Z_RESULT_STEPS - Z_BORDER_PIX) * 
        Z_BLOCKDIM_X + threadIdx.x;
    const int r0 = blockIdx.y * Z_BLOCKDIM_Y + threadIdx.y;
    const int c0 = blockIdx.z + threadIdx.z;

    in  += c0 * sze_z * sze_r + r0 * sze_z + z0;
    out += c0 * sze_z * sze_r + r0 * sze_z + z0;

    {pragma_unroll:s}
    for (int i = Z_BORDER_PIX; i < Z_BORDER_PIX + Z_RESULT_STEPS; i++) {{
        shr_data[threadIdx.y][threadIdx.x+i*Z_BLOCKDIM_X] = 
            (sze_z - z0 > i * Z_BLOCKDIM_X) ? in[i*Z_BLOCKDIM_X] : 0;
    }}
    {pragma_unroll:s}
    for (int i = 0; i < Z_BORDER_PIX; i++) {{
        shr_data[threadIdx.y][threadIdx.x+i*Z_BLOCKDIM_X] = 
            (z0 >= -i*Z_BLOCKDIM_X) ? in[i*Z_BLOCKDIM_X] : 0;
    }}
    {pragma_unroll:s}
    for (int i = Z_BORDER_PIX + Z_RESULT_STEPS; i < 2*Z_BORDER_PIX+Z_RESULT_STEPS; i++) {{
        shr_data[threadIdx.y][threadIdx.x+i*Z_BLOCKDIM_X] =
            (sze_z - z0 > i * Z_BLOCKDIM_X) ? in[i*Z_BLOCKDIM_X] : 0;
    }}
    
    __syncthreads();
    if (r0 >= sze_r) {{
        return;
    }}
    {pragma_unroll:s}
    for (int i = Z_BORDER_PIX; i < Z_BORDER_PIX+Z_RESULT_STEPS; i++) {{
        if (sze_z - z0 > i*Z_BLOCKDIM_X) {{
            float sum = 0.0f;
            {pragma_unroll:s}
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {{
                sum += kernel[KERNEL_RADIUS-j] * 
                    (float)shr_data[threadIdx.y][threadIdx.x+i*Z_BLOCKDIM_X+j];
            }}
            out[i*Z_BLOCKDIM_X] = sum;
        }}
    }}
}}


template<typename T>
__global__ void convRKernel(float* out, T* in, float* kernel,
                            const int sze_z, const int sze_r, const int sze_c)
{{
    __shared__ T shr_data[R_BLOCKDIM_X][(R_RESULT_STEPS+2*R_BORDER_PIX)
                                         *R_BLOCKDIM_Y+1];
    const int z0 = blockIdx.x * R_BLOCKDIM_X + threadIdx.x;
    const int r0 = (blockIdx.y * R_RESULT_STEPS - R_BORDER_PIX) * 
        R_BLOCKDIM_Y + threadIdx.y;
    const int c0 = blockIdx.z + threadIdx.z;

    in  += c0 * sze_z * sze_r + r0 * sze_z + z0;
    out += c0 * sze_z * sze_r + r0 * sze_z + z0;

    {pragma_unroll:s}
    for (int i = R_BORDER_PIX; i < R_BORDER_PIX + R_RESULT_STEPS; i++) {{
        shr_data[threadIdx.x][threadIdx.y+i*R_BLOCKDIM_Y] = 
            (sze_r - r0 > i * R_BLOCKDIM_Y) ? in[i*R_BLOCKDIM_Y*sze_z] : 0;
    }}
    {pragma_unroll:s}
    for (int i = 0; i < R_BORDER_PIX; i++) {{
        shr_data[threadIdx.x][threadIdx.y+i*R_BLOCKDIM_Y] = 
            (r0 >= -i*R_BLOCKDIM_Y) ? in[i*R_BLOCKDIM_Y*sze_z] : 0;
    }}
    {pragma_unroll:s}
    for (int i = R_BORDER_PIX + R_RESULT_STEPS; i < 2*R_BORDER_PIX+R_RESULT_STEPS; i++) {{
        shr_data[threadIdx.x][threadIdx.y+i*R_BLOCKDIM_Y] =
            (sze_r - r0 > i * R_BLOCKDIM_Y) ? in[i*R_BLOCKDIM_Y*sze_z] : 0;
    }}

    __syncthreads();
    if (z0 >= sze_z) {{
        return;
    }}
    {pragma_unroll:s}
    for (int i = R_BORDER_PIX; i < R_BORDER_PIX + R_RESULT_STEPS; i++) {{
        if (sze_r - r0 > i * R_BLOCKDIM_Y) {{
            float sum = 0.0f;
            {pragma_unroll:s}
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {{
                sum += kernel[KERNEL_RADIUS-j] * 
                    (float)shr_data[threadIdx.x][threadIdx.y+i*R_BLOCKDIM_Y+j];
            }}
            out[i*R_BLOCKDIM_Y*sze_z] = sum;
        }}
    }}
}}


template<typename T>
__global__ void convCKernel(float* out, T* in, float* kernel,
                            const int sze_z, const int sze_r, const int sze_c)
{{
    __shared__ T shr_data[C_BLOCKDIM_X][(C_RESULT_STEPS+2*C_BORDER_PIX)
                                         *C_BLOCKDIM_Z+1];
    const int z0 = blockIdx.x * C_BLOCKDIM_X + threadIdx.x;
    const int r0 = blockIdx.y + threadIdx.y;
    const int c0 = (blockIdx.z * C_RESULT_STEPS - C_BORDER_PIX) * 
        C_BLOCKDIM_Z + threadIdx.z;
    
    in  += c0 * sze_z * sze_r + r0 * sze_z + z0;
    out += c0 * sze_z * sze_r + r0 * sze_z + z0;

    {pragma_unroll:s}
    for (int i = C_BORDER_PIX; i < C_BORDER_PIX + C_RESULT_STEPS; i++) {{
        shr_data[threadIdx.x][threadIdx.z+i*C_BLOCKDIM_Z] = 
            (sze_c - c0 > i * C_BLOCKDIM_Z) ? in[i*C_BLOCKDIM_Z*sze_z*sze_r] : 0;
    }}
    for (int i = 0; i < C_BORDER_PIX; i++) {{
        shr_data[threadIdx.x][threadIdx.z+i*C_BLOCKDIM_Z] = 
            (c0 >= -i * C_BLOCKDIM_Z) ? in[i*C_BLOCKDIM_Z*sze_z*sze_r] : 0;
    }}
    for (int i = C_BORDER_PIX + C_RESULT_STEPS; i < 2*C_BORDER_PIX+C_RESULT_STEPS; i++) {{
        shr_data[threadIdx.x][threadIdx.z+i*C_BLOCKDIM_Z] =
            (sze_c - c0 > i * C_BLOCKDIM_Z) ? in[i*C_BLOCKDIM_Z*sze_z*sze_r] : 0;
    }}

    __syncthreads();
    if (z0 >= sze_z) {{
        return;
    }}
    {pragma_unroll:s}
    for (int i = C_BORDER_PIX; i < C_BORDER_PIX + C_RESULT_STEPS; i++) {{
        if (sze_c - c0 > i * C_BLOCKDIM_Z) {{
            float sum = 0.0f;
            {pragma_unroll:s}
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {{
                sum += kernel[KERNEL_RADIUS-j] * 
                    (float)shr_data[threadIdx.x][threadIdx.z+i*C_BLOCKDIM_Z+j];
            }}
            out[i*C_BLOCKDIM_Z*sze_z*sze_r] = sum;
        }}
    }}
}}