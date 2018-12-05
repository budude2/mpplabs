#include <stdlib.h>
#include "support.h"
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

void startTime(Timer* timer)
{
        gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer)
{
        gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer)
{
        return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

std::vector<cv::Mat> loadFiles(char **argv)
{

    std::vector<cv::Mat> img;
    unsigned int index = 0;

    boost::filesystem::path p(argv[1]);

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

                for (boost::filesystem::directory_entry &x : boost::filesystem::directory_iterator(p))
                {
                    if(is_regular_file(x.path()))
                    {
                        std::cout << "Loading: " << x.path().string() << std::endl;

                        img.push_back(cv::imread(x.path().string()));

                        if (!img[index].data)
                        {
                            std::cin.get();
                            //return -1;
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

    catch (const boost::filesystem::filesystem_error &ex)
    {
        std::cout << ex.what() << std::endl;
    }

    return img;
}