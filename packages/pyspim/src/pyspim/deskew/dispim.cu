__global__ void deskewTexture(float* deskewed, cudaTextureObject_t texObj,
                              int width, int height, int depth, 
                              double rel_pixel_size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        for (int z = 0; z < depth; z++) {
            float x2 = (float)x * rel_pixel_size - z;
            deskewed[x + width*y + z*width*height] = tex3D<float>(
                texObj, x2/rel_pixel_size, y+0.5f, z+0.5f
            );
        }
    }
}

template<typename T>
__global__ void rotateYAxisPositive(T* in, T* out, 
                                    size_t width, size_t height, size_t depth)
{
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i < width && j < height && k < depth) {
        size_t oidx = (width - i - 1)*height*depth + j*depth + k;
        size_t iidx = k*height*width + j*width + i;
        out[oidx] = in[iidx];
    }
}


template<typename T>
__global__ void rotateYAxisNegative(T* in, T* out, 
                                    size_t width, size_t height, size_t depth)
{
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const size_t k = blockDim.z * blockIdx.z + threadIdx.z;
    if (i < width && j < height && k < depth) {
        size_t oidx = i * height * depth + j * depth + (depth - k - 1);
        size_t iidx = k*height*width + j*width + i;
        out[oidx] = in[iidx];
    }
}
