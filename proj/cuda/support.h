#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>
#include <opencv2/opencv.hpp>

typedef struct
{
    struct timeval startTime;
    struct timeval endTime;
} Timer;


void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
std::vector<cv::Mat> loadFiles(char **argv);

#endif
