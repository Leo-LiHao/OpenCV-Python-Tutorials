#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '18/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np


def getGaussianPyramid(Img, levelNum):
    Pyramid = [Img]
    for level in xrange(levelNum-1):
        Pyramid.append(cv2.pyrDown(Pyramid[level]))
    return Pyramid

def getLaplacianPyramid(img, levelNum):
    GaussianPyramid = getGaussianPyramid(img, levelNum+1)
    LaplacianPyramid = []
    for i in xrange(levelNum):
        LaplacianPyramid.append(cv2.subtract(GaussianPyramid[i], cv2.pyrUp(GaussianPyramid[i+1])))
    return LaplacianPyramid


if __name__ == '__main__':
    Img = cv2.imread('../../../Datas/lena.png')
    # ImgDown = cv2.pyrDown(Img)
    ImgPyramid = getGaussianPyramid(Img, 5)
    for i, img in enumerate(ImgPyramid):
        print i
        cv2.imshow('GuassianPyramid'+str(i), img)
    cv2.imshow('Img', Img)
    # cv2.imshow('ImgDown', ImgDown)
    cv2.waitKey()

    cv2.destroyAllWindows()
    ImgPyramid = getLaplacianPyramid(Img, 5)
    for i, img in enumerate(ImgPyramid):
        print i
        cv2.namedWindow('LaplacianPyramid'+str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('LaplacianPyramid'+str(i), img)
    cv2.imshow('Img', Img)
    cv2.waitKey()