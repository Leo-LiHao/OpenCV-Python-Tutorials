#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '15/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np


def AverangeFilter(img, kernelSize):
    assert len(kernelSize) == 2, 'kernelSize must contain 2 number'
    Kernel = np.ones((kernelSize[0], kernelSize[1]), dtype=np.float32) / (kernelSize[0] * kernelSize[1])
    return cv2.filter2D(src=img, ddepth=-1, kernel=Kernel)


if __name__ == '__main__':
    Img = cv2.imread('../../../Datas/OpencvLogo.png')
    Kernel = np.ones((5,5), dtype=np.float32) / 25
    DstImg = cv2.filter2D(src=Img, ddepth=-1, kernel=Kernel)
    cv2.imshow('SrcImg', Img)
    cv2.imshow('DstImg', DstImg)
    cv2.waitKey()
