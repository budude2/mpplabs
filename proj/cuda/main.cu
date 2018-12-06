#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"
#include "support.h"
#include <cuda.h>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: avg [path to images]\n";
        return 1;
    }

    Timer timer;
    std::vector<cv::Mat> img;
    std::vector<cv::Mat> channels(3);
    cudaError_t cuda_ret;

    cudaStream_t blueStream, greenStream, redStream;
    cudaStreamCreate(&blueStream);
    cudaStreamCreate(&greenStream);
    cudaStreamCreate(&redStream);

    /**********************************
        LOAD IMAGES
    **********************************/
    startTime(&timer);

    img = loadFiles(argv);

    stopTime(&timer);
    std::cout << "\nLoad images.........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Setup & allocation
    **********************************/
    startTime(&timer);

    const unsigned int numRows   = img[0].rows;
    const unsigned int numCols   = img[0].cols;
    const unsigned int stepSize  = img[0].step;
    const unsigned int vecSize   = img.size();            // Number of images loaded

    const unsigned long long imageSize        = stepSize * numRows;    // Size of the image in bytes
    const unsigned long long channelSize      = imageSize / 3;
    const unsigned long long channelBlockSize = channelSize * vecSize;

    cv::Mat res(numRows, numCols, CV_8UC3);
    cv::Mat blueres(numRows, numCols, CV_8UC1);
    cv::Mat greenres(numRows, numCols, CV_8UC1);
    cv::Mat redres(numRows, numCols, CV_8UC1);

    const unsigned int resSize        = res.step * res.rows;
    const unsigned int resChannelSize = resSize / 3;

    unsigned char *blueData;
    cuda_ret = cudaMallocHost((void**)&blueData, channelBlockSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *greenData;
    cuda_ret = cudaMallocHost((void**)&greenData, channelBlockSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *redData;
    cuda_ret = cudaMallocHost((void**)&redData, channelBlockSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *blueResData;
    cuda_ret = cudaMallocHost((void**)&blueResData, resChannelSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *greenResData;
    cuda_ret = cudaMallocHost((void**)&greenResData, resChannelSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *redResData;
    cuda_ret = cudaMallocHost((void**)&redResData, resChannelSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    unsigned char *blue_d, *green_d, *red_d;
    unsigned char *resblue_d, *resgreen_d, *resred_d;

    cuda_ret = cudaMalloc<unsigned char>(&blue_d, channelBlockSize);
    cuda_ret = cudaMalloc<unsigned char>(&green_d, channelBlockSize);
    cuda_ret = cudaMalloc<unsigned char>(&red_d, channelBlockSize);
    cuda_ret = cudaMalloc<unsigned char>(&resblue_d, resChannelSize);
    cuda_ret = cudaMalloc<unsigned char>(&resgreen_d, resChannelSize);
    cuda_ret = cudaMalloc<unsigned char>(&resred_d, resChannelSize);
    
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Sync error") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -1;
    }

    stopTime(&timer);
    std::cout << "Setup & allocation..." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Data setup
    **********************************/
    startTime(&timer);

    for(unsigned int i = 0; i < vecSize; i++)
    {
        split(img[i], channels);
        memcpy(&blueData[channelSize * i], channels[0].data, channelSize);
        memcpy(&greenData[channelSize * i], channels[1].data, channelSize);
        memcpy(&redData[channelSize * i], channels[2].data, channelSize);
    }

    stopTime(&timer);
    std::cout << "Data setup..........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Run kernel
    **********************************/
    startTime(&timer);

    const dim3 block(32,32);
    const dim3 grid((numCols + block.x - 1)/block.x, (numRows + block.y - 1)/block.y);

    cudaMemcpyAsync(blue_d, blueData, channelBlockSize, cudaMemcpyHostToDevice, blueStream);
    cudaMemcpyAsync(green_d, greenData, channelBlockSize, cudaMemcpyHostToDevice, greenStream);
    cudaMemcpyAsync(red_d, redData, channelBlockSize, cudaMemcpyHostToDevice, redStream);

    blue_proc<<<grid, block, 0, blueStream>>>(blue_d, resblue_d, numCols, numRows, stepSize/3, channelSize, vecSize);
    green_proc<<<grid, block, 0, greenStream>>>(green_d, resgreen_d, numCols, numRows, stepSize/3, channelSize, vecSize);
    red_proc<<<grid, block, 0, redStream>>>(red_d, resred_d, numCols, numRows, stepSize/3, channelSize, vecSize);

    cudaMemcpyAsync(blueResData, resblue_d, resChannelSize, cudaMemcpyDeviceToHost, blueStream);
    cudaMemcpyAsync(greenResData, resgreen_d, resChannelSize, cudaMemcpyDeviceToHost, greenStream);
    cudaMemcpyAsync(redResData, resred_d, resChannelSize, cudaMemcpyDeviceToHost, redStream);

    cudaDeviceSynchronize();

    stopTime(&timer);
    std::cout << "Run kernel..........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Merge data
    **********************************/

    startTime(&timer);

    memcpy(blueres.ptr(), blueResData, resChannelSize);
    memcpy(greenres.ptr(), greenResData, resChannelSize);
    memcpy(redres.ptr(), redResData, resChannelSize);

    std::vector<cv::Mat> resChannels;
    resChannels.push_back(blueres);
    resChannels.push_back(greenres);
    resChannels.push_back(redres);

    cv::merge(resChannels, res);

    stopTime(&timer);
    std::cout << "Merge data..........." << elapsedTime(timer) << " s" << std::endl;

    startTime(&timer);

    std::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);

    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << "Write data..........." << elapsedTime(timer) << " s" << std::endl;

    // stopTime(&timer);
    // std::cout << elapsedTime(timer) << " s" << std::endl;
    
    cudaFreeHost(blue_d);
    cudaFreeHost(green_d);
    cudaFreeHost(red_d);
    cudaFreeHost(resblue_d);
    cudaFreeHost(resgreen_d);
    cudaFreeHost(resred_d);
    res.release();

    // cudaProfilerStop();
    // cudaDeviceReset();

    return 0;
}
