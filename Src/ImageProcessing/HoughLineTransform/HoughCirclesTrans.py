#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '23/04/2015'


import sys
sys.path.append("../../../")

import cv2
import cv2.cv as cv
import time
import math
import numpy as np

import Src.ToolBox.ImageProcessTool     as IPT


if __name__ == '__main__':
    SrcImg = cv2.imread('../../../Datas/OpencvLogo.png')
    SrcImg = cv2.GaussianBlur(src=SrcImg, ksize=(7, 7), sigmaX=1)
    GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    CannyThreshHigh = 50
    CannyThreshLow = 30
    Circles = cv2.HoughCircles(image=GrayImg, method=cv.CV_HOUGH_GRADIENT,
                               dp=1, minDist=20, param1=CannyThreshHigh,
                               param2=CannyThreshLow, minRadius=0, maxRadius=0)

    Circles = np.uint16(np.around(Circles))
    for circle in Circles[0,:]:
        Center = (circle[0], circle[1])
        Radius = circle[2]
        # draw the outer circle
        cv2.circle(SrcImg, Center, Radius, (0,0,255), 2)
        # draw the center of the circle
        cv2.circle(SrcImg, Center, 2, (0,255,0), 3)

    cv2.imshow('detected circles',SrcImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()