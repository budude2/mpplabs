#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "support.h"

int main()
{
    Timer timer;

    std::cout << "Opening images......";
    startTime(&timer);

    cv::Mat image1, image2, image3, image4, image5;

    image1 = cv::imread("../noise_set2/noise1.jpg", 1);
    image2 = cv::imread("../noise_set2/noise2.jpg", 1);
    image3 = cv::imread("../noise_set2/noise3.jpg", 1);
    image4 = cv::imread("../noise_set2/noise4.jpg", 1);
    image5 = cv::imread("../noise_set2/noise5.jpg", 1);

    if(!image1.data)
    {
        std::cout << "Could not open image 1" << std::endl;
        return -1;
    }
    if(!image2.data)
    {
        std::cout << "Could not open image 2" << std::endl;
        return -1;
    }
    if(!image3.data)
    {
        std::cout << "Could not open image 3" << std::endl;
        return -1;
    }
    if(!image4.data)
    {
        std::cout << "Could not open image 4" << std::endl;
        return -1;
    }
    if(!image5.data)
    {
        std::cout << "Could not open image 5" << std::endl;
        return -1;
    }

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Processing images...";
    startTime(&timer);

    image1.convertTo(image1, CV_32F);
    image2.convertTo(image2, CV_32F);
    image3.convertTo(image3, CV_32F);
    image4.convertTo(image4, CV_32F);
    image5.convertTo(image5, CV_32F);

    unsigned int width = image1.cols;
    unsigned int height = image1.rows;

    cv::Mat res(height, width, CV_32F);

    res = (image1 + image2 + image3 + image4 + image5) / 5;

    res.convertTo(res, CV_8U);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Writing image.......";
    startTime(&timer);

    cv::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);

    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    image1.release();
    image2.release();
    image3.release();
    image4.release();
    image5.release();
    res.release();
    return 0;
}
