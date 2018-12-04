#include <iostream>
#include <opencv2/opencv.hpp>
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

    std::cout << "Images loaded..." << std::endl;
    std::cout << "Vector size: " << img.size() << std::endl;

    stopTime(&timer);
    std::cout << "Opening images......";
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Processing images...";
    startTime(&timer);

    unsigned int width  = img[0].cols;
    unsigned int height = img[0].rows;

    cv::Mat res(height, width, CV_32FC3);

    // Convert images to 3 channel float and find the sum.
    for(unsigned int i = 0; i < img.size(); i++)
    {
        img[i].convertTo(img[i], CV_32F);
        res += img[i];
    }

    // Take the average
    res = res / img.size();

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Writing image.......";
    startTime(&timer);

    // Write the resulting image.
    res.convertTo(res, CV_8U);

    std::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);
    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    // for(unsigned int i = 0; i < img.size(); i++)
    // {
    //     img[i].release();
    // }

    // res.release();

    // delete &img;

    return 0;
}