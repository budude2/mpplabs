#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"
#include "support.h"
#include <cuda.h>
#include <cuda_profiler_api.h>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: avg [path to images]\n";
        return 1;
    }

    std::vector<cv::Mat> img;
    Timer timer;
    cudaError_t cuda_ret;

    /**********************************
        LOAD IMAGES
    **********************************/
    startTime(&timer);

    img = loadFiles(argv);

    stopTime(&timer);
    std::cout << "\nOpening images.....";
    std::cout << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Setup & allocation
    **********************************/
    startTime(&timer);

    unsigned int numRows   = img[0].rows;
    unsigned int numCols   = img[0].cols;
    unsigned int stepSize  = img[0].step;
    unsigned int vecSize   = img.size();            // Number of images loaded

    unsigned long long imageSize = stepSize * numRows;    // Size of the image in bytes
    unsigned long long blockSize = imageSize * vecSize;   // Total utilization of all images

    cv::Mat res(numRows, numCols, CV_8UC3);

    unsigned char *imageData;
    cuda_ret = cudaMallocHost((void**)&imageData, blockSize);
    if (cuda_ret != cudaSuccess)
        printf("Error allocating pinned host memory\n");

    const int resSize   = res.step * res.rows;

    unsigned char *res_d;
    unsigned char *img_d;

    cuda_ret = cudaMalloc<unsigned char>(&img_d, blockSize);
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Unable to allocate memory") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -1;
    }

    cuda_ret = cudaMalloc<unsigned char>(&res_d, resSize);
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Unable to allocate memory") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -1;
    }

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Sync error") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -1;
    }

    stopTime(&timer);
    std::cout << "Setup & allocate..." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Data setup
    **********************************/
    startTime(&timer);

    for(unsigned int i = 0; i < vecSize; i++)
    {
        memcpy(&imageData[imageSize * i], img[i].data, imageSize);
    }
    
    stopTime(&timer);
    std::cout << "Data setup........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Data copy
    **********************************/   
    startTime(&timer);

    cuda_ret = cudaMemcpy(img_d, imageData, blockSize, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Unable to copy data") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -2;
    }
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Sync error") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -2;
    }

    stopTime(&timer);
    std::cout << "Data copy.........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Run kernel
    **********************************/
    startTime(&timer);

    const dim3 block(16,16);
    const dim3 grid((numCols + block.x - 1)/block.x, (numRows + block.y - 1)/block.y);

    image_proc<<<grid, block>>>(img_d, res_d, numCols, numRows, stepSize, imageSize, vecSize);

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess){
        std::cerr << ("Unable to launch kernel") << std::endl;
        std::cerr << "Error code: " << cuda_ret << std::endl;
        return -2;
    }

    stopTime(&timer);
    std::cout << "Run kernel........." << elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Copy result
    **********************************/
    startTime(&timer);

    cudaMemcpy(res.ptr(), res_d, resSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    stopTime(&timer);
    std::cout << "Copy result........" <<elapsedTime(timer) << " s" << std::endl;

    /**********************************
        Copy result
    **********************************/
    startTime(&timer);

    std::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);

    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << "Write result......." << elapsedTime(timer) << " s" << std::endl;

    cudaFreeHost(imageData);
    res.release();
    cudaFree(img_d);
    cudaFree(res_d);

    return 0;
}
