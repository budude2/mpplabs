#!/usr/bin/env python3

import cv2
import numpy
import time

image0 = cv2.imread('noise1.jpg', cv2.IMREAD_COLOR)
image1 = cv2.imread('noise2.jpg', cv2.IMREAD_COLOR)
image2 = cv2.imread('noise3.jpg', cv2.IMREAD_COLOR)
image3 = cv2.imread('noise4.jpg', cv2.IMREAD_COLOR)
image4 = cv2.imread('noise5.jpg', cv2.IMREAD_COLOR)

numRow = image0.shape[0];
numCol = image0.shape[1];

meanImg = numpy.zeros((numRow, numCol, 3), numpy.uint8)

start_time = time.time()

for col in range(0, numCol - 1):
    for row in range(0, numRow - 1):

        # Get average of the blue channel
        blueAvg = numpy.mean([image0.item(row, col, 0), image1.item(row, col, 0), image2.item(row, col, 0), image3.item(row, col, 0), image4.item(row, col, 0)]).astype(numpy.uint8)

        # Store average into blue channel
        meanImg.itemset((row, col, 0), blueAvg)
        
        # Get average of the green channel
        greenAvg = numpy.mean([image0.item(row, col, 1), image1.item(row, col, 1), image2.item(row, col, 1), image3.item(row, col, 1), image4.item(row, col, 1)]).astype(numpy.uint8)

        # Store average into green channel
        meanImg.itemset((row, col, 1), greenAvg)

        # Get average of the red channel
        redAvg = numpy.mean([image0.item(row, col, 2), image1.item(row, col, 2), image2.item(row, col, 2), image3.item(row, col, 2), image4.item(row, col, 2)]).astype(numpy.uint8)

        # Store average into red channel
        meanImg.itemset((row, col, 2), redAvg)

print("--- %s seconds ---" % (time.time() - start_time))


cv2.imwrite('mean.jpg', meanImg)
