#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "kernel.cu"
#include "support.h"

int main()
{
    Timer timer;

    std::cout << "Opening images..................";
    startTime(&timer);
    
    cv::Mat image1;
    cv::Mat image2;
    cv::Mat image3;
    cv::Mat image4;
    cv::Mat image5;

    image1 = cv::imread("noise1.jpg", 1);
    image2 = cv::imread("noise2.jpg", 1);
    image3 = cv::imread("noise3.jpg", 1);
    image4 = cv::imread("noise4.jpg", 1);
    image5 = cv::imread("noise5.jpg", 1);

    if(!image1.data)
    {
        std::cout << "Could not open image" << std::endl;
    }
    if(!image2.data)
    {
        std::cout << "Could not open image" << std::endl;
    }
    if(!image3.data)
    {
        std::cout << "Could not open image" << std::endl;
    }
    if(!image4.data)
    {
        std::cout << "Could not open image" << std::endl;
    }
    if(!image5.data)
    {
        std::cout << "Could not open image" << std::endl;
    }

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    cv::Mat res(image1.rows, image1.cols, CV_8UC3);

    const int imageSize = image1.step * image1.rows;
    const int resSize   = res.step * res.rows;

    unsigned char *d_image1;
    unsigned char *d_image2;
    unsigned char *d_image3;
    unsigned char *d_image4;
    unsigned char *d_image5;
    unsigned char *d_res;

    std::cout << "Allocating memory on device.....";
    startTime(&timer);

    cudaMalloc<unsigned char>(&d_image1, imageSize);
    cudaMalloc<unsigned char>(&d_image2, imageSize);
    cudaMalloc<unsigned char>(&d_image3, imageSize);
    cudaMalloc<unsigned char>(&d_image4, imageSize);
    cudaMalloc<unsigned char>(&d_image5, imageSize);
    cudaMalloc<unsigned char>(&d_res, resSize);
    
    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;
    
    std::cout << "Copying images to device........";
    startTime(&timer);
    
    cudaMemcpy(d_image1, image1.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image2, image2.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image3, image3.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image4, image4.ptr(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image5, image5.ptr(), imageSize, cudaMemcpyHostToDevice);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    const dim3 block(16,16);
    const dim3 grid((image1.cols + block.x - 1)/block.x, (image1.rows + block.y - 1)/block.y);

    std::cout << "Running kernel..................";
    startTime(&timer);

    image_proc<<<grid, block>>>(d_image1, d_image2, d_image3, d_image4, d_image5, d_res, image1.cols, image1.rows, image1.step, res.step);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Copying result from device......";
    startTime(&timer);

    cudaMemcpy(res.ptr(), d_res, resSize, cudaMemcpyDeviceToHost);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Write image to HD...............";
    startTime(&timer);

    cv::imwrite("result.jpg", res);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    image1.release();
    image2.release();
    image3.release();
    image4.release();
    image5.release();
    res.release();
    cudaFree(d_image1);
    cudaFree(d_image2);
    cudaFree(d_image3);
    cudaFree(d_image4);
    cudaFree(d_image5);
    cudaFree(d_res);
}
