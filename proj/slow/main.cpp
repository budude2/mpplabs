#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "support.h"

namespace cv
{
    using std::vector;
}

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

    unsigned int width = image1.cols;
    unsigned int height = image1.rows;

    cv::Mat res(height, width, CV_32FC3);

    image1.convertTo(image1, CV_32F);
    image2.convertTo(image2, CV_32F);
    image3.convertTo(image3, CV_32F);
    image4.convertTo(image4, CV_32F);
    image5.convertTo(image5, CV_32F);

    for(unsigned int col = 0; col < width; col++)
    {
       for(unsigned int row = 0; row < height; row++)
       {
            cv::Vec3f pixel0 = image1.at<cv::Vec3f>(row, col);
            cv::Vec3f pixel1 = image2.at<cv::Vec3f>(row, col);
            cv::Vec3f pixel2 = image3.at<cv::Vec3f>(row, col);
            cv::Vec3f pixel3 = image4.at<cv::Vec3f>(row, col);
            cv::Vec3f pixel4 = image5.at<cv::Vec3f>(row, col);

            float blueAvg  = (pixel0.val[0] + pixel1.val[0] + pixel2.val[0] + pixel3.val[0] + pixel4.val[0]) / 5;
            float greenAvg = (pixel0.val[1] + pixel1.val[1] + pixel2.val[1] + pixel3.val[1] + pixel4.val[1]) / 5;
            float redAvg   = (pixel0.val[2] + pixel1.val[2] + pixel2.val[2] + pixel3.val[2] + pixel4.val[2]) / 5;

            res.at<cv::Vec3f>(row, col)[0] = blueAvg;
            res.at<cv::Vec3f>(row, col)[1] = greenAvg;
            res.at<cv::Vec3f>(row, col)[2] = redAvg;
       }
    }

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
