#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '30/04/2016'


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

def drawCircleBoard(imgSize, boardSize, circleDis, circleRadius, originPt, backgroundColor, foregroundColor):
    Canvas = np.zeros(imgSize, dtype=np.uint8)
    Canvas[:, :] = backgroundColor
    RowNum, ColNum = boardSize
    OriginPoint_2x1 = np.array(originPt).reshape(2, 1)
    Circles_2xn = np.array([0, 0]).reshape(2, 1)
    for row in xrange(RowNum):
        for col in xrange(ColNum):
            x = col * circleDis
            y = row * circleDis
            Point = np.array([[x],
                              [y]])
            Circles_2xn = np.hstack((Circles_2xn, Point))
    Circles_2xn = Circles_2xn[:, 1:] + OriginPoint_2x1
    IPT.drawPoints(img=Canvas, pts_2xn=Circles_2xn, color=foregroundColor, radius=circleRadius)
    return Canvas


if __name__ == '__main__':
    CircleImg = \
        drawCircleBoard(imgSize=(500, 500, 3), boardSize=(7, 7), circleDis=50,
                        circleRadius=int(50*0.3), originPt=(50, 50), backgroundColor=(0, 0, 0),
                        foregroundColor=(255, 255, 255))
    # CircleImg = \
    #     drawCircleBoard(imgSize=(500, 500), boardSize=(7, 7), circleDis=50,
    #                     circleRadius=int(50*0.3), originPt=(50, 50), backgroundColor=0,
    #                     foregroundColor=255)

    FilePath = '../../../Datas/Chessboard.jpg'
    # SrcImg = cv2.imread(FilePath)
    SrcImg = CircleImg
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)

    # ------------------- SIFT ------------------- #
    SIFT = cv2.SIFT(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
    KeyPoints = SIFT.detect(GrayImg)
    KeyPoints, Descriptions = SIFT.compute(GrayImg, KeyPoints)
    # another way
    T0 = time.time()
    KeyPoints_2, Descriptions2 = SIFT.detectAndCompute(GrayImg, mask=None)
    UseTime = time.time() - T0
    print '------------------------ SIFT ------------------------ '
    print 'Usetime:', UseTime
    print 'Keypoints: ', len(KeyPoints), 'Descriptions: ', Descriptions.shape
    SIFTShowImg = cv2.drawKeypoints(image=GrayImg, keypoints=KeyPoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ------------------- SURF ------------------- #
    # HessianThreshold = 400          # who is larger, retain. 300 ~ 500 is good
    HessianThreshold = 50000        # for demo
    GaussianPyramidNumber = 4
    OctaveLayers = 2                # The number of images within each octave of a gaussian pyramid.
    DimDescription_64or128 = True   # False - 64,  True - 128
    useU_SURF = False                  # Up-right SURF flag, U-SURF have not orientation
    SURF = cv2.SURF(hessianThreshold=HessianThreshold, nOctaves=GaussianPyramidNumber,
                    nOctaveLayers=OctaveLayers, extended=DimDescription_64or128, upright=useU_SURF)
    T0 = time.time()
    KeyPoints, Descriptions = SURF.detectAndCompute(GrayImg, mask=None)
    UseTime = time.time() - T0
    print '------------------------ SURF ------------------------ '
    print 'Usetime:', UseTime
    print 'Keypoints: ', len(KeyPoints), 'Descriptions: ', Descriptions.shape
    SURFShowImg = cv2.drawKeypoints(image=GrayImg, keypoints=KeyPoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # ------------------- U-SURF ------------------- #
    useU_SURF = True
    U_SURF = cv2.SURF(hessianThreshold=HessianThreshold, nOctaves=GaussianPyramidNumber,
                    nOctaveLayers=OctaveLayers, extended=DimDescription_64or128, upright=useU_SURF)
    T0 = time.time()
    KeyPoints, Descriptions = U_SURF.detectAndCompute(GrayImg, mask=None)
    UseTime = time.time() - T0
    print '------------------------ U-SURF ------------------------ '
    print 'Usetime:', UseTime
    print 'Keypoints: ', len(KeyPoints), 'Descriptions: ', Descriptions.shape
    USURFShowImg = cv2.drawKeypoints(image=GrayImg, keypoints=KeyPoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    FeaturePts_2xn = np.array([0, 0]).reshape(2, 1)
    for Pt in KeyPoints:
        x, y = Pt.pt
        FeaturePts_2xn = np.hstack((FeaturePts_2xn, np.array([x, y]).reshape(2, 1)))
    FeaturePts_2xn = FeaturePts_2xn[:, 1:]
    IPT.drawPoints(img=SrcImg, pts_2xn=FeaturePts_2xn, color=(0, 0, 255), radius=3)
    cv2.imshow('Src', SrcImg)
    cv2.imshow('SIFT', SIFTShowImg)
    cv2.imshow('U-SURF', USURFShowImg)
    cv2.imshow('SURF', SURFShowImg)
    cv2.waitKey()

