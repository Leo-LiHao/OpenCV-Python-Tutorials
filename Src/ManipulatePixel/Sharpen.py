#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '26/01/2015'


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def Sharpen2D(img):
    Kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    NewImg = cv2.filter2D(src=img, ddepth=-1, kernel=Kernel)
    return NewImg


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/lena.png')
    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
    SharpenImg = Sharpen2D(img=Img)
    SharpenGrayImg = Sharpen2D(img=GrayImg)
    cv2.imshow('SharpenImg', SharpenImg)
    cv2.imshow('SharpenGrayImg', SharpenGrayImg)
    cv2.imshow('SrcImg', Img)
    cv2.imshow('GrayImg', GrayImg)
    cv2.waitKey()