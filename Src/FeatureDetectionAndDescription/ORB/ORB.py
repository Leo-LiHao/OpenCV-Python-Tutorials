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

    FeaturePointsNum = 500
    PyramidScale = 1.2
    PyramidLevel = 8
    PatchSize = 31           # size of the patch used by the oriented BRIEF descriptor
    EdgeThresh = PatchSize
    FirstLevel = 0           # default
    OrientedBRIEFPointNum = 2  # 2, 3, 4
    ScoreType = cv2.ORB_HARRIS_SCORE
    # ScoreType = cv2.ORB_FAST_SCORE

    ORB = cv2.ORB(nfeatures=FeaturePointsNum, scaleFactor=PyramidScale, nlevels=PyramidLevel, edgeThreshold=EdgeThresh,
                  firstLevel=FirstLevel, WTA_K=OrientedBRIEFPointNum, scoreType=ScoreType, patchSize=PatchSize)
    KeyPoints = ORB.detect(GrayImg, mask=None)
    KeyPoints, Descriptions = ORB.compute(GrayImg, KeyPoints)
    ShowImg = cv2.drawKeypoints(image=SrcImg, keypoints=KeyPoints, color=RED)

    print 'Key points number:', len(KeyPoints)
    print Descriptions.shape
    cv2.imshow('Result', ShowImg)
    cv2.waitKey()
