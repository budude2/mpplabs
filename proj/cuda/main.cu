#include <opencv2/opencv.hpp>
#include <iostream>
#include "kernel.cu"
#include "support.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: avg [path to images]\n";
        return 1;
    }

    std::vector<cv::Mat> img;
    Timer timer;

    startTime(&timer);

    img = loadFiles(argv);

    unsigned int numRows   = img[0].rows;
    unsigned int numCols   = img[0].cols;
    unsigned int stepSize  = img[0].step;
    unsigned int vecSize   = img.size();            // Number of images loaded

    unsigned int imageSize = stepSize * numRows;    // Size of the image in bytes
    unsigned int blockSize = imageSize * vecSize;   // Total utilization of all images

    cv::Mat res(numRows, numCols, CV_8UC3);

    unsigned char *imageData = new unsigned char[blockSize];

    std::cout << "\nOpening images....";
    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    startTime(&timer);
    for(unsigned int i = 0; i < vecSize; i++)
    {
        memcpy(&imageData[imageSize * i], img[i].data, imageSize);
    }

    std::cout << "Images -> block...";
    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Allocate arrays...";
    startTime(&timer);

    const int resSize   = res.step * res.rows;

    unsigned char *res_d;
    unsigned char *img_d;

    cudaMalloc<unsigned char>(&img_d, blockSize);
    cudaMalloc<unsigned char>(&res_d, resSize);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Copy images.......";
    startTime(&timer);

    cudaMemcpy(img_d, imageData, blockSize, cudaMemcpyHostToDevice);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Launch kernel.....";
    startTime(&timer);

    const dim3 block(16,16);
    const dim3 grid((numCols + block.x - 1)/block.x, (numRows + block.y - 1)/block.y);

    image_proc<<<grid, block>>>(img_d, res_d, numCols, numRows, stepSize, imageSize, vecSize);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Copy result.......";
    startTime(&timer);

    cudaMemcpy(res.ptr(), res_d, resSize, cudaMemcpyDeviceToHost);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Write result......";
    startTime(&timer);

    std::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);

    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    delete[] imageData;
    res.release();
    cudaFree(img_d);
    cudaFree(res_d);
}
