#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '25/04/2016'


import sys
sys.path.append('../../../')

import cv2
import numpy as np

import Src.ToolBox.ImageProcessTool     as IPT


def markContours(contous, markImg, startNum):
    markNum = startNum
    for contour in contous:
        cv2.drawContours(markImg, [contour], 0, color=markNum, thickness=-1)
        markNum += 1
    return markNum, markImg



if __name__ == '__main__':
    SrcImg = cv2.imread('../../../Datas/Coins.jpg')
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    _, BinImg = cv2.threshold(src=GrayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    # filter
    Kernel = np.ones((3, 3), dtype=np.uint8)
    OpenImg = cv2.morphologyEx(src=BinImg, op=cv2.MORPH_OPEN, kernel=Kernel, iterations=2)
    cv2.imshow('OpenImg', OpenImg)

    # background
    Kernel = np.ones((5, 5), dtype=np.uint8)
    DilationImg = cv2.morphologyEx(src=OpenImg, op=cv2.MORPH_DILATE, kernel=Kernel, iterations=1)
    cv2.imshow('DilationImg', DilationImg)

    # foreground
    DistImg = cv2.distanceTransform(src=OpenImg, distanceType=cv2.cv.CV_DIST_L2, maskSize=5)
    C = 255.0 / DistImg.max()
    DistNorm = np.uint8(DistImg*C)
    cv2.imshow('DistNorm', DistNorm)
    _, DistBin = cv2.threshold(src=DistNorm, thresh=DistNorm.max()*0.7, maxval=255, type=cv2.THRESH_BINARY)
    cv2.imshow('DistBin', DistBin)
    # cv2.imshow('DistImg', cv2.normalize(src=DistImg, norm_type=cv2.NORM_MINMAX))

    # unknown
    # Unknown = DilationImg - DistBin
    Unknown = cv2.subtract(DilationImg, DistBin)
    cv2.imshow('Unknown', Unknown)

    # connect
    MarkImg = np.zeros(GrayImg.shape, dtype=np.int32)
    MarkImg[DilationImg == 0] = 1
    ForegroundContours, _ = cv2.findContours(image=DistBin.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    ret, MarkImg = markContours(ForegroundContours, markImg=MarkImg, startNum=2)
    cv2.imshow('MarkImg', np.uint8(MarkImg * (255.0 / MarkImg.max())))

    cv2.watershed(SrcImg, MarkImg)
    SrcImg[MarkImg == -1] = [0, 0, 255]
    cv2.imshow('Src', SrcImg)
    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Bin', BinImg)
    cv2.waitKey()