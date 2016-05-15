#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '02/05/2016'


import sys
sys.path.append('../../../')

import cv2
import time
import numpy as np

from Src.ImageProcessing.Contours.ContourAnalyst import ContourAnalyst
import Src.ToolBox.ImageProcessTool     as IPT


RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)


if __name__ == '__main__':
    FilePath = '../../../Datas/Chessboard.jpg'
    # FilePath = '../../../Datas/SmallChessboard.png'
    SrcImg = cv2.imread(FilePath)
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    FAST = cv2.FastFeatureDetector(threshold=50, nonmaxSuppression=True)
    KeyPoints = FAST.detect(GrayImg, mask=None)
    ShowImg = cv2.drawKeypoints(image=SrcImg, keypoints=KeyPoints, color=RED)

    FAST.setBool('nonmaxSuppression', False)
    KeyPoints2 = FAST.detect(GrayImg, mask=None)
    ShowImg2 = cv2.drawKeypoints(image=SrcImg, keypoints=KeyPoints2, color=RED)

    cv2.imshow('Result_nonmaxSuppression', ShowImg)
    cv2.imshow('Result', ShowImg2)
    cv2.waitKey()
