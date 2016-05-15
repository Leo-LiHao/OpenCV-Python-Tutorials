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
    Img = cv2.imread('../../../Datas/lena.png')
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2))
    T0 = time.time()
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    print 'cv2.INTER_CUBIC: ', time.time() - T0
    cv2.imshow('INTER_CUBIC', ResizeImg)

    T0 = time.time()
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv2.INTER_LANCZOS4)
    print 'cv2.INTER_LANCZOS4: ', time.time() - T0
    cv2.imshow('INTER_LANCZOS4', ResizeImg)

    T0 = time.time()
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv2.INTER_AREA)
    print 'cv2.INTER_AREA: ', time.time() - T0
    cv2.imshow('INTER_AREA', ResizeImg)

    T0 = time.time()
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv2.INTER_LINEAR)
    print 'cv2.INTER_LINEAR: ', time.time() - T0
    cv2.imshow('INTER_LINEAR', ResizeImg)

    T0 = time.time()
    ResizeImg = cv2.resize(src=Img, dsize=(Img.shape[1]*2, Img.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    print 'cv2.INTER_NEAREST: ', time.time() - T0
    cv2.imshow('INTER_NEAREST', ResizeImg)

    cv2.imshow('Src', Img)
    cv2.waitKey()
