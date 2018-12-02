__global__ void image_proc(unsigned char* image1, unsigned char* image2, unsigned char* image3, unsigned char* image4, unsigned char* image5, unsigned char* output, int width, int height, int colorSize, int layerSize)
{
    const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex < width) & (yIndex < height))
    {
        const int image_idx  = yIndex * colorSize + (3 * xIndex);
        const int res_idx    = yIndex * layerSize + (3 * xIndex);

        const unsigned char blue_avg  = (image1[image_idx] + image2[image_idx] + image3[image_idx] + image4[image_idx] + image5[image_idx]) / 5;
        const unsigned char green_avg = (image1[image_idx + 1] + image2[image_idx + 1] + image3[image_idx + 1] + image4[image_idx + 1] + image5[image_idx + 1]) / 5;
        const unsigned char red_avg   = (image1[image_idx + 2] + image2[image_idx + 2] + image3[image_idx + 2] + image4[image_idx + 2] + image5[image_idx + 2]) / 5;

        output[res_idx]     = static_cast<unsigned char>(blue_avg);
        output[res_idx + 1] = static_cast<unsigned char>(green_avg);
        output[res_idx + 2] = static_cast<unsigned char>(red_avg);
    }
}
