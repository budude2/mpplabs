__global__ void blue_proc(unsigned char* images, unsigned char* output, unsigned int width, unsigned int height, unsigned int colorSize, unsigned long long imageSize, unsigned int numImg)
{
    const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex < width) & (yIndex < height))
    {
        const unsigned int image_idx  = yIndex * colorSize + xIndex;
        const unsigned int res_idx    = yIndex * colorSize + xIndex;

        __shared__ float blue_avg;
        blue_avg = 0;
        unsigned char blue_avg_char  = 0;

        for(unsigned int imgNum = 0; imgNum < numImg; imgNum++)
        {
            blue_avg  += images[imgNum * imageSize + image_idx];
        }

        blue_avg_char  = blue_avg   / numImg;
        output[res_idx]     = static_cast<unsigned char>(blue_avg_char);
    }
}

__global__ void green_proc(unsigned char* images, unsigned char* output, unsigned int width, unsigned int height, unsigned int colorSize, unsigned long long imageSize, unsigned int numImg)
{
    const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex < width) & (yIndex < height))
    {
        const unsigned int image_idx  = yIndex * colorSize + xIndex;
        const unsigned int res_idx    = yIndex * colorSize + xIndex;

        __shared__ float green_avg;
        green_avg = 0;
        unsigned char green_avg_char = 0;

        for(unsigned int imgNum = 0; imgNum < numImg; imgNum++)
        {
            green_avg += images[imgNum * imageSize + image_idx];
        }

        green_avg_char = green_avg  / numImg;

        output[res_idx] = static_cast<unsigned char>(green_avg_char);
    }
}

__global__ void red_proc(unsigned char* images, unsigned char* output, unsigned int width, unsigned int height, unsigned int colorSize, unsigned long long imageSize, unsigned int numImg)
{
    const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if((xIndex < width) & (yIndex < height))
    {
        const unsigned int image_idx  = yIndex * colorSize + xIndex;
        const unsigned int res_idx    = yIndex * colorSize + xIndex;

        __shared__ float red_avg;
        red_avg = 0;
        unsigned char red_avg_char   = 0;

        for(unsigned int imgNum = 0; imgNum < numImg; imgNum++)
        {
            red_avg   += images[imgNum * imageSize + image_idx];
        }

        red_avg_char   = red_avg    / numImg;

        output[res_idx] = static_cast<unsigned char>(red_avg_char);
    }
}