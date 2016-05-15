#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '23/04/2015'


import sys
sys.path.append("../../../")

import cv2
import time
import math
import numpy as np

import Src.ToolBox.ImageProcessTool     as IPT


def HoughLineTrans(edgeImg, rhoStep, thetaStep, thresh):
    cv2.imshow('Edge', edgeImg)
    MaxRho = int(np.linalg.norm(np.array([edgeImg.shape[0], edgeImg.shape[1]])))+1
    RhoIter = np.arange(-MaxRho-rhoStep, MaxRho+rhoStep, rhoStep)
    ThetaRange = np.arange(0, np.pi, thetaStep)
    CosVal = np.cos(ThetaRange)
    SinVal = np.sin(ThetaRange)
    CosSinMatrix_2xn = np.vstack((CosVal, SinVal))
    HoughImg = np.zeros((len(RhoIter), len(ThetaRange)), dtype=np.uint8)
    Loc = np.nonzero(edgeImg)
    AllPts_Px2 = np.hstack((Loc[1].reshape(-1, 1), Loc[0].reshape(-1, 1)))
    CalcResult = np.dot(AllPts_Px2, CosSinMatrix_2xn)
    Ymap = np.repeat(np.arange(len(ThetaRange), dtype=np.int32).reshape(1, -1), CalcResult.shape[0], 0)
    for x, y in zip(CalcResult.ravel(), Ymap.ravel()):
        if abs(x-x.round()) < 1e-9:
            XCoord = int(x.round()) - (-MaxRho-rhoStep) / int(rhoStep)
            HoughImg[XCoord, y] += 1
    C = 255 / HoughImg.max()
    HoughImg = C * HoughImg
    cv2.namedWindow('HoughImg', cv2.WINDOW_NORMAL)
    cv2.imshow('HoughImg', HoughImg)
    Rho, Theta = np.where(HoughImg>thresh)
    return np.hstack((RhoIter[Rho].reshape(-1, 1), (Theta*thetaStep).reshape(-1, 1)))


if __name__ == '__main__':
    # SrcImg = cv2.imread('../../../Datas/PeopleInShadow.jpg')
    # SrcImg = cv2.imread('../../../Datas/Edge.png')
    SrcImg = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.line(img=SrcImg, pt1=(100, 100), pt2=(100, 200), color=(255,255,255), thickness=1)
    GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
    Edges = cv2.Canny(GrayImg, 50, 150, apertureSize = 3)
    Lines = cv2.HoughLines(image=Edges, rho=1, theta=np.pi/180, threshold=50)
    DrawImg = SrcImg.copy()
    print '--------------------- CV ---------------------'
    cv2.namedWindow('HoughLines_CV', cv2.WINDOW_NORMAL)
    for rho, theta in Lines[0]:
        print 'rho:  ', rho
        print 'theta:', theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img=DrawImg, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=1)
        print (x1, y1), (x2, y2)
    cv2.imshow('HoughLines_CV', DrawImg)

    Lines = HoughLineTrans(edgeImg=Edges, rhoStep=1, thetaStep=np.pi/180, thresh=5)
    DrawImg = SrcImg.copy()
    cv2.namedWindow('HoughLines', cv2.WINDOW_NORMAL)
    print '--------------------- DIY ---------------------'
    for rho, theta in Lines:
        print 'rho:  ', rho
        print 'theta:', theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img=DrawImg, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=1)
        print (x1, y1), (x2, y2)
    cv2.imshow('HoughLines', DrawImg)
    cv2.waitKey()