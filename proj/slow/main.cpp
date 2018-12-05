#include <opencv2/opencv.hpp>
#include <iostream>
#include "support.h"

int main(int argc, char* argv[])
{
    Timer timer;
    std::vector<cv::Mat> img;

    //std::cout << "Opening images......";
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
    unsigned int numImages = img.size();

    cv::Mat res(height, width, CV_8UC3);

    unsigned long long blueAvg  = 0;
    unsigned long long greenAvg = 0;
    unsigned long long redAvg   = 0;

    for(unsigned int col = 0; col < width; col++)
    {
       for(unsigned int row = 0; row < height; row++)
       {
            blueAvg  = 0;
            greenAvg = 0;
            redAvg   = 0;

            for(unsigned int imgNum = 0; imgNum < numImages; imgNum++)
            {
                cv::Vec3b pixel = img[imgNum].at<cv::Vec3b>(row, col);

                blueAvg  += pixel.val[0];
                greenAvg += pixel.val[1];
                redAvg   += pixel.val[2];

                res.at<cv::Vec3b>(row, col)[0] = static_cast<unsigned char>(blueAvg  / numImages);
                res.at<cv::Vec3b>(row, col)[1] = static_cast<unsigned char>(greenAvg / numImages);
                res.at<cv::Vec3b>(row, col)[2] = static_cast<unsigned char>(redAvg   / numImages);
            }
       }
    }

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    std::cout << "Writing image.......";
    startTime(&timer);

    std::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);

    cv::imwrite("result.jpg", res, compression_param);

    stopTime(&timer);
    std::cout << elapsedTime(timer) << " s" << std::endl;

    res.release();

    return 0;
}
