#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct
{
        struct timeval startTime;
            struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
        void startTime(Timer* timer);
            void stopTime(Timer* timer);
                float elapsedTime(Timer timer);
#ifdef __cplusplus
}
#endif

#endif
