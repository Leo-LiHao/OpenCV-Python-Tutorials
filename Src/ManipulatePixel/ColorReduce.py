#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '26/01/2015'


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def ColorReduce(img, div=64):
    divFactor = int(div)
    NewImg = img / divFactor * divFactor + divFactor / 2
    return NewImg


if __name__ == '__main__':
    Img = cv2.imread('../Datas/lena.png')
    reducedImg = ColorReduce(img=Img, div=8)
    cv2.imshow('reducedImg', reducedImg)
    cv2.imshow('SrcImg', Img)
    cv2.waitKey()