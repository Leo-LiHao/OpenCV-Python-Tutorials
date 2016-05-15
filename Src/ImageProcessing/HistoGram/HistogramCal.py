#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/02/2015'


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def calcHistogram_np(grayImg):
    hist, bins = np.histogram(grayImg.flatten(), bins=256, range=[0,256], normed=False)
    pass

def calcHistogram(img, channel=0, mask=None, histSize=256, ranges=(0, 256)):
    Hist = cv2.calcHist(images=[img], channels=[channel], mask=mask, histSize=[histSize], ranges=ranges)
    return Hist

def calcHistogramCDF(img, channel=0, mask=None, histSize=256, ranges=(0, 256), nomalize=False):
    Hist = cv2.calcHist(images=[img], channels=[channel], mask=mask, histSize=[histSize], ranges=ranges)
    CDF = Hist.cumsum()
    if nomalize:
        HistCumNormalized = CDF / CDF.max() * Hist.max()
        return HistCumNormalized
    return CDF

def drawHistImg(hist, color=(255,255,255)):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3], np.uint8)
    hpt = 0.9 * 256
    for h in range(256):
        intensity = int(hist[h]*hpt/maxVal)
        cv2.line(histImg, (h,256), (h,256-intensity), color)
    return histImg

def showHistogram_matplot(img, figure=None):
    if figure is not None:
        Figure = figure
    else:
        Figure = plt.figure()
    if img.ndim == 2:
        plt.hist(img.ravel(), 256, [0,256])
    elif img.ndim == 3:
        Colors = ('b','g','r')
        for idx, color in enumerate(Colors):
            Hist = cv2.calcHist([img], [idx], None, [256], [0, 256])
            plt.plot(Hist, color=color)
            plt.xlim([0, 256])
    else:
        raise ValueError, 'img must be 2 or 3 dimension'
    return Figure


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/lena.png')
    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
    Hist = calcHistogram_np(GrayImg)
    Histogram = calcHistogram(img=Img, channel=0)
    HistImg = drawHistImg(hist=Histogram)
    cv2.imshow('HistImg', HistImg)
    GrayHistogram = calcHistogram(img=GrayImg, channel=0)
    GrayHistImg = drawHistImg(hist=GrayHistogram)
    cv2.imshow('Gray-HistImg', GrayHistImg)
    Figure = showHistogram_matplot(img=Img)
    plt.show(Figure)
    # showHistogram_matplot(img=GrayImg)
    # TestImg3D = np.ones((2,2,2), dtype=np.uint8) * 254
    # TestImg2D = np.ones((2,2), dtype=np.uint8) * 255
    # Hist = cv2.calcHist(images=[Img], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    # Hist3D = cv2.calcHist(images=[TestImg3D], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    # Hist2D = cv2.calcHist(images=[TestImg2D], channels=[0], mask=None, histSize=[256], ranges=[0.0, 255.0])
    # print Hist2D