__global__ void image_proc(unsigned char* images, unsigned char* output, unsigned int width, unsigned int height, unsigned int colorSize, unsigned long long imageSize, unsigned int numImg)
{
    const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex < width) & (yIndex < height))
    {
        const unsigned int image_idx  = yIndex * colorSize + (3 * xIndex);
        const unsigned int res_idx    = yIndex * colorSize + (3 * xIndex);

        float blue_avg = 0;
        float green_avg = 0;
        float red_avg = 0;
        unsigned char blue_avg_char = 0;
        unsigned char green_avg_char = 0;
        unsigned char red_avg_char = 0;

        for(unsigned int imgNum = 0; imgNum < numImg; imgNum++)
        {
            blue_avg  += images[imgNum * imageSize + image_idx];
            green_avg += images[imgNum * imageSize + image_idx + 1];
            red_avg   += images[imgNum * imageSize + image_idx + 2];
        }

        blue_avg_char = blue_avg / numImg;
        green_avg_char = green_avg / numImg;
        red_avg_char = red_avg / numImg;

        output[res_idx]     = static_cast<unsigned char>(blue_avg_char);
        output[res_idx + 1] = static_cast<unsigned char>(green_avg_char);
        output[res_idx + 2] = static_cast<unsigned char>(red_avg_char);
    }
}
