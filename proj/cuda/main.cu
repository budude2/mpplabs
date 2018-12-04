#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "kernel.cu"

int main()
{
    std::vector<cv::Mat> img;

    img.push_back(cv::imread("../noise_set2/noise1.jpg", 1));
    img.push_back(cv::imread("../noise_set2/noise2.jpg", 1));
    img.push_back(cv::imread("../noise_set2/noise3.jpg", 1));
    img.push_back(cv::imread("../noise_set2/noise4.jpg", 1));
    img.push_back(cv::imread("../noise_set2/noise5.jpg", 1));

    cv::Mat res(img[0].rows, img[0].cols, CV_8UC3);

    unsigned int vecSize = img.size();
    unsigned int imageSize = img[0].step * img[0].rows;
    unsigned int blockSize = imageSize * vecSize;

    unsigned char **images = new unsigned char*[vecSize];
    unsigned char *imageData = new unsigned char[blockSize];

    for(unsigned int i = 0; i < vecSize; i++)
    {
        images[i] = img[i].data;
    }

    for(unsigned int j = 0; j < vecSize; j++)
    {
        for(unsigned int i = 0; i < imageSize; i++)
        {
            imageData[j * imageSize + i] = images[j][i];
        }
    }

    std::cout << sizeof(imageData) / sizeof(*imageData) << std::endl;

    const int resSize   = res.step * res.rows;

    unsigned char *d_res;
    unsigned char *img_d;

    cudaMalloc<unsigned char>(&img_d, blockSize);
    cudaMalloc<unsigned char>(&d_res, resSize);

    cudaMemcpy(img_d, imageData, blockSize, cudaMemcpyHostToDevice);

    const dim3 block(16,16);
    const dim3 grid((img[0].cols + block.x - 1)/block.x, (img[0].rows + block.y - 1)/block.y);

    image_proc<<<grid, block>>>(img_d, d_res, img[0].cols, img[0].rows, img[0].step, img[0].step * img[0].rows, img.size());

    cudaMemcpy(res.ptr(), d_res, resSize, cudaMemcpyDeviceToHost);

    cv::imwrite("result.jpg", res);

    delete[] images;
    delete[] imageData;
    res.release();
    cudaFree(img_d);
    cudaFree(d_res);
}
