CC = g++
CFLAGS = -g -Wall
SRCS = main.cpp support.cpp
PROG = avg

OPENCV = -lopencv_core -lopencv_highgui -lopencv_imgcodecs
BOOST = -lboost_filesystem -lboost_system
LIBS = $(OPENCV) $(BOOST)

$(PROG):$(SRCS)
		$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
