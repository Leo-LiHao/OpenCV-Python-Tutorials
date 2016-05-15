#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '28/04/2016'


import sys
sys.path.append('../../../')

import cv2
import numpy as np

from Src.ImageProcessing.Contours.ContourAnalyst import ContourAnalyst
import Src.ToolBox.ImageProcessTool     as IPT


RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)


if __name__ == '__main__':
    FilePath = '../../../Datas/Chessboard.jpg'
    # FilePath = '../../../Datas/Edge.png'
    SrcImg = cv2.imread(FilePath)
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    # Corners_nx1x2 = cv2.goodFeaturesToTrack(image=GrayImg, maxCorners=54, qualityLevel=0.01,
    #                                   minDistance=10, blockSize=5, useHarrisDetector=True, k=0.04)

    # ----------------------- Formula ----------------------- #
    # Shi-Tomasi Corner Detector
    # R = min(lambda_1, lambda_2)
    # ----------------------- Formula ----------------------- #
    Corners_nx1x2 = cv2.goodFeaturesToTrack(image=GrayImg, maxCorners=150, qualityLevel=0.01, minDistance=15)
    # Corners_nx1x2 = cv2.goodFeaturesToTrack(image=GrayImg, maxCorners=4, qualityLevel=0.01, minDistance=15)
    print 'Corners number: ', Corners_nx1x2.shape[0]
    Corners_2xn = Corners_nx1x2.T.reshape(2, -1)
    IPT.drawPoints(img=SrcImg, pts_2xn=Corners_2xn, color=RED, radius=3)
    cv2.imshow('Src', SrcImg)
    cv2.waitKey()
