#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '13/04/2015'


import sys
sys.path.append("../../../")

import cv2
import time
import cv2.cv as cv
import numpy as np



if __name__ == '__main__':
    Img = cv2.imread('../../Datas/lena.png')
    AffineMatrix = np.array([[1, 0, 100],
                             [0, 1,  50]], dtype=np.float32)
    DstImg = cv2.warpAffine(Img, AffineMatrix, (Img.shape[0]*2, Img.shape[1]*2), borderValue=(155, 155, 155))
    cv2.imshow('Src', Img)
    cv2.imshow('DstImg', DstImg)
    RotateMatrix = cv2.getRotationMatrix2D(center=(Img.shape[1]/2, Img.shape[0]/2),
                                           angle=90,
                                           scale=2)
    print '(Img.shape[1]/2, Img.shape[0]/2):', (Img.shape[1]/2, Img.shape[0]/2)
    print 'Rotate Matrix:'
    print RotateMatrix
    RotImg = cv2.warpAffine(Img, RotateMatrix, (Img.shape[0]*2, Img.shape[1]*2))
    cv2.imshow('rotate', RotImg)
    cv2.waitKey()
