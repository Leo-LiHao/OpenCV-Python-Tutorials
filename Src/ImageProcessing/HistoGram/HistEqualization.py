#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/02/2015'


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import HistogramCal      as hc


def localEqualizeHist(src, size):
    assert 'uint8' == src.dtype

    Res = src.copy()
    if 2 == src.ndim:
        Row, Col = src.shape
        for i in xrange(Row - size[0]):
            for j in xrange(Col - size[1]):
                SrcRoi = src[i:i+size[0], j:j+size[1]]
                ResRoi = Res[i:i+size[0], j:j+size[1]]
                ResRoi[size[0]/2, size[1]/2] = cv2.equalizeHist(src=SrcRoi)[size[0]/2, size[1]/2]
        return Res
    elif 3 == src.ndim:
        B = localEqualizeHist(src=src[:,:,0], size=size)
        G = localEqualizeHist(src=src[:,:,1], size=size)
        R = localEqualizeHist(src=src[:,:,2], size=size)
        return cv2.merge([B, G, R])
    

if __name__ == '__main__':
    # Img = cv2.imread('../../Datas/Logo.jpg')
    # Img = cv2.imread('../../Datas/PeopleInShadow.jpg')
    Img = cv2.imread('../../../Datas/InsertMachine/SrcA.bmp')
    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)[:400,:400]
    T0 = time.time()
    CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # CLAHE = cv2.createCLAHE(clipLimit=1000)
    DstImg = CLAHE.apply(GrayImg)
    T1 = time.time()

    EquImg = cv2.equalizeHist(GrayImg)
    T2 = time.time()

    HKHImg = localEqualizeHist(GrayImg, (15, 15))
    T3 = time.time()
    print 'image size: ', GrayImg.shape
    print 'use time: '
    print 'EqualizeHist ', T2 - T1
    print 'CLAHE        ', T1 - T0
    print 'HKH          ', T3 - T2
    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Equ', EquImg)
    cv2.imshow('Dst', DstImg)
    cv2.imshow('HKHImg', HKHImg)
    cv2.waitKey()

# if __name__ == '__main__':
#     Img = cv2.imread('../../Datas/Logo.jpg')
#     GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
#     cv2.imshow('GrayImg', GrayImg)
#     EqualImg = cv2.equalizeHist(src=GrayImg)
#     cv2.imshow('EqualImg', EqualImg)
#
#     Hist = hc.calcHistogram(GrayImg, 0)
#     HistCumNormalized_pure = hc.calcHistogramCDF(Hist)
#     Figure_pure = hc.showHistogram_matplot(GrayImg)
#     plt.plot(HistCumNormalized_pure, color='k', figure=Figure_pure)
#     plt.xlim([0, 256])
#
#     HistEqu = hc.calcHistogram(EqualImg, 0)
#     HistCumNormalized_equ = hc.calcHistogramCDF(HistEqu)
#     Figure_equ = hc.showHistogram_matplot(EqualImg)
#     plt.plot(HistCumNormalized_equ, color='k', figure=Figure_equ)
#     plt.xlim([0, 256])
#
#     NewImg = EqualImg
#     NewImg[NewImg<150] = 0
#     cv2.imshow('new', NewImg)
#     HistNew = hc.calcHistogram(NewImg, 0)
#     HistCumNormalized_new = hc.calcHistogramCDF(HistNew)
#     Figure_new = hc.showHistogram_matplot(NewImg)
#     plt.plot(HistCumNormalized_new, color='k', figure=Figure_new)
#     plt.xlim([0, 256])
#     NewEquImg = cv2.equalizeHist(src=NewImg)
#     cv2.imshow('NewEquImg', NewEquImg)
#
#     plt.show()
