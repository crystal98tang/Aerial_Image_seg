# -*- coding: utf-8 -*-
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256], np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1

    return grayHist


if __name__ == "__main__":
    #if len(sys.argv) > 1:
    #    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    #else:
    #    print("sys.argv null")
    image = cv2.imread("G:/Aerial_Image_seg_v1.0/test_img/test_img.jpg", cv2.IMREAD_GRAYSCALE)

    grayHist = calcGrayHist(image)

    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=1, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue])
    plt.xlabel(U"灰度值")
    plt.ylabel(U"像素数")
    plt.show()

