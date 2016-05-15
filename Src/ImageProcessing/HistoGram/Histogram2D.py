#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '21/04/2016'


import sys
sys.path.append('../../../')

import cv2
import numpy as np
from   matplotlib import pyplot as plt

import Src.ToolBox.ImageProcessTool as IPT
import HistogramCal


DownPoint = [0,0]
Roi_xyxy = np.array([0,0,1,1])

def OnMouse(event, x, y, flags, *args):
    global DownPoint
    global Roi_xyxy

    if cv2.EVENT_LBUTTONDOWN == event:
        DownPoint = [x, y]
    elif cv2.EVENT_LBUTTONUP == event:
        if DownPoint[0] != x and DownPoint[1] != y:
            Roi_xyxy[0:2] = [min(DownPoint[0], x), min(DownPoint[1], y)]
            Roi_xyxy[2:4] = [max(DownPoint[0], x), max(DownPoint[1], y)]
            print 'roi_xyxy:', Roi_xyxy
            print 'roi_xywh:', IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)
    else:
        return


def calcHist2D(img):
    HSVImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    Hist = cv2.calcHist(images=[HSVImg], channels=[0, 1], mask=None, histSize=[180, 256], ranges=[0, 180, 0, 256])
    return Hist

def calcHist2D_np(img):
    HSVImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    H = HSVImg[:, :, 0]
    S = HSVImg[:, :, 1]
    Hist, xBins, yBins = np.histogram2d(H.ravel(), S.ravel(),[180,256],[[0,180],[0,256]])
    return Hist, xBins, yBins

def show2DHist_plt(hist):
    plt.imshow(hist, interpolation='nearest')
    plt.show()

def backProject_np(matchImg, modelImg, thresh=0.5):
    MatchHist = calcHist2D(matchImg)
    MatchHSV = cv2.cvtColor(src=matchImg, code=cv2.COLOR_BGR2HSV)
    ModelHist = calcHist2D(modelImg)
    H = matchImg[:, :, 0]
    S = matchImg[:, :, 1]
    Ratio = ModelHist / MatchHist
    Probability = Ratio[H.ravel(), S.ravel()]
    Probability = np.minimum(Probability, 1)
    Probability[Probability>thresh] = 255
    Probability = Probability.reshape(MatchHSV.shape[:2]).astype(np.uint8)
    return Probability

if __name__ == '__main__':
    cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Src", OnMouse)
    while True:
        Img = cv2.imread('../../../Datas/Paper3.jpg')
        HSVImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2HSV)
        _, RoiImg = IPT.getRoiImg(img=Img, roi=Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY, copy=True)

        RoiHist2D = calcHist2D(RoiImg)
        RoiHist2D = cv2.normalize(src=RoiHist2D, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # DstImg = cv2.calcBackProject(images=[HSVImg], channels=[0, 1], hist=RoiHist2D, ranges=[0, 180, 0, 256], scale=1)
        DstImg = backProject_np(HSVImg, RoiImg)
        cv2.imshow('Back', DstImg)
        Disc = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5,5))
        DstImg = cv2.filter2D(src=DstImg, ddepth=-1, kernel=Disc)
        cv2.imshow('Filter2D', DstImg)

        _, MaskImg = cv2.threshold(src=DstImg, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
        cv2.imshow('Mask', MaskImg)
        Mask3D = cv2.merge((MaskImg, MaskImg, MaskImg))
        ResultImg = cv2.bitwise_and(src1=Img, src2=Mask3D)
        cv2.imshow('ResultImg', ResultImg)

        IPT.drawRoi(img=Img, roi=Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY, color=(0, 0, 255))
        cv2.imshow('Src', Img)
        cv2.imshow('Roi', RoiImg)

        Key = chr(cv2.waitKey(15) & 255)
        if Key == 'q':
            break