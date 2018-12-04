CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp support.cpp
PROG = avg_st

OPENCV = -lopencv_core -lopencv_highgui
LIBS = $(OPENCV)

$(PROG):$(SRCS)
		$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

slow:avg_st_slow.cpp
		$(CC) $(CFLAGS) -o avg_st_slow avg_st_slow.cpp $(LIBS)

