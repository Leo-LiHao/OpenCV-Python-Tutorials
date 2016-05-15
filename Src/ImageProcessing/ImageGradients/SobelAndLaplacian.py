#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '16/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np


def normalize(img):
    AbsImg = np.abs(img)
    Scale = AbsImg.max() / 255
    NormalizedImg_float = AbsImg / Scale
    assert NormalizedImg_float.max() < 256, 'error'
    return np.uint8(NormalizedImg_float)


if __name__ == '__main__':
    # GrayImg = np.zeros((760, 1024), dtype=np.uint8)
    # GrayImg[200:500, 200:700] = 255
    GrayImg = cv2.imread('../../../Datas/lena.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    KernelSize = 3
    # Depth = -1
    Depth = cv2.CV_64F
    LaplacianImg = cv2.Laplacian(src=GrayImg, ddepth=Depth, ksize=KernelSize)
    SobelX = cv2.Sobel(src=GrayImg, ddepth=Depth, dx=1, dy=0, ksize=KernelSize)
    SobelY = cv2.Sobel(src=GrayImg, ddepth=Depth, dx=0, dy=1, ksize=KernelSize)
    print 'SobelX max: ', np.abs(SobelX).max()
    print 'SobelX mean: ', np.abs(SobelX[SobelX>0]).mean()

    # ---------------------------- Scharr ---------------------------- #
    ScharrX = cv2.Sobel(src=GrayImg, ddepth=Depth, dx=1, dy=0, ksize=-1)
    ScharrY = cv2.Sobel(src=GrayImg, ddepth=Depth, dx=0, dy=1, ksize=-1)
    print 'ScharrX max: ', np.abs(ScharrX).max()
    print 'ScharrX mean: ', np.abs(ScharrX[SobelX>0]).mean()

    # print np.absolute(SobelX)
    # print SobelX[0,1]
    # print LaplacianImg
    ScharrX_norm = normalize(ScharrX)
    ScharrY_norm = normalize(ScharrY)
    SobelX_norm = normalize(SobelX)
    SobelY_norm = normalize(SobelY)
    Laplacian_norm = normalize(LaplacianImg)

    cv2.imshow('Laplacian', Laplacian_norm)
    cv2.imshow('SobelX', SobelX_norm)
    cv2.imshow('SobelY', SobelY_norm)
    cv2.imshow('ScharrX_norm', ScharrX_norm)
    cv2.imshow('ScharrY_norm', ScharrY_norm)
    cv2.imshow('Src', GrayImg)

    # ---------------------------- Canny ---------------------------- #
    SobelKernelSize = 3
    # threshold1 : threshold2   =    1 : 3   is a better param
    CannyImg = cv2.Canny(image=GrayImg, threshold1=655/10, threshold2=655/10*7, apertureSize=SobelKernelSize, L2gradient=True)
    # CannyImg = cv2.Canny(image=GrayImg, threshold1=100, threshold2=1081, apertureSize=SobelKernelSize, L2gradient=True)
    # CannyImg = cv2.Canny(image=GrayImg, threshold1=100, threshold2=1082, apertureSize=SobelKernelSize, L2gradient=True)
    cv2.imshow('Canny', CannyImg)
    cv2.waitKey()