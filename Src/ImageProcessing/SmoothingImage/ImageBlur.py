#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '15/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np



if __name__ == '__main__':
    Img = cv2.imread('../../../Datas/OpencvLogo.png')
    cv2.imshow('Img', Img)
    # ---------------------------- Blur ---------------------------- #
    KernelSize = (5, 5)
    BlurImg = cv2.blur(src=Img, ksize=KernelSize)
    cv2.imshow('cvBlur', BlurImg)
    KernelSize = (5, 5)
    BoxFilterImg = cv2.boxFilter(src=Img, ddepth=-1, ksize=KernelSize, normalize=True)
    cv2.imshow('cvbox', BoxFilterImg)
    print np.equal(BlurImg, BoxFilterImg).all()
    cv2.waitKey()

    # ---------------------------- Gaussian Blur ---------------------------- #
    KernelSize = (15, 15)
    GaussianBlurImg = cv2.GaussianBlur(src=Img, ksize=KernelSize, sigmaX=0, sigmaY=0)

    # KernelSize = (5, 15)
    GaussianKernelX = cv2.getGaussianKernel(ksize=KernelSize[0], sigma=0)
    GaussianKernelY = cv2.getGaussianKernel(ksize=KernelSize[1], sigma=0)
    print 'GaussianKernelX:\n', GaussianKernelX
    print 'GaussianKernelY:\n', GaussianKernelY
    FilterGaussian = cv2.sepFilter2D(src=Img, ddepth=-1, kernelX=GaussianKernelX, kernelY=GaussianKernelY)
    cv2.imshow('Gauss', GaussianBlurImg)
    cv2.imshow('FilterGaussian', FilterGaussian)
    print np.equal(GaussianBlurImg, FilterGaussian).all()
    cv2.waitKey()

    # ---------------------------- Median Blur ---------------------------- #
    KernelSize = 5
    MeanBlurImg = cv2.medianBlur(src=Img, ksize=KernelSize)
    cv2.imshow('MeanBlurImg', MeanBlurImg)
    cv2.waitKey()

    # ---------------------------- Bilateral Blur ---------------------------- #
    BilateralBlur = cv2.bilateralFilter(src=Img, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imshow('BilateralBlur', BilateralBlur)
    cv2.waitKey()