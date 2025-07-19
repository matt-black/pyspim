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