#include <iostream>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace boost::filesystem;

namespace cv
{
    using std::vector;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: test path\n";
        return 1;
    }

    unsigned int index = 0;
    cv::vector<cv::Mat> img;
    path p(argv[1]);

    // Load all images from the specified folder
    try
    {
        if (exists(p))
        {
            if (is_regular_file(p))
            {
                std::cout << "Please give a directory." << std::endl;
            }
            else if (is_directory(p))
            {
                std::cout << "Loading images from: " << p << "\n" << std::endl;

                for (directory_entry &x : directory_iterator(p))
                {
                    if(is_regular_file(x.path()))
                    {
                        std::cout << "Loading: " << x.path().string() << std::endl;

                        img.push_back(cv::imread(x.path().string()));

                        if (!img[index].data)
                        {
                            std::cin.get();
                            return -1;
                        }

                        index++;
                    }
                }
            }
            else
            {
                std::cout << p << " exists, but is not a regular file or directory" << std::endl;
            }
        }
        else
        {
            std::cout << p << " does not exist" << std::endl;
        }
    }

    catch (const filesystem_error &ex)
    {
        std::cout << ex.what() << std::endl;
    }

    std::cout << "Images loaded..." << std::endl;
    std::cout << "Vector size: " << img.size() << std::endl;

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

    // Write the resulting image.
    res.convertTo(res, CV_8U);

    cv::vector<int> compression_param;
    compression_param.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_param.push_back(100);
    cv::imwrite("result.jpg", res, compression_param);

    return 0;
}