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

import Src.ToolBox.ImageProcessTool as IPT


LeftPoint = [0, 0]
RightPoint = [0, 0]

def OnMouse1(event, x, y, flags, *args):
    global LeftPoint

    if cv2.EVENT_LBUTTONDOWN == event:
        LeftPoint = [x, y]
    else:
        return

def OnMouse2(event, x, y, flags, *args):
    global RightPoint
    if cv2.EVENT_RBUTTONDOWN == event:
        RightPoint = [x, y]
    else:
        return


if __name__ == '__main__':
    First = True
    SrcImg = cv2.imread('../../../Datas/AffineLena.png')
    SrcCanvas = np.zeros(SrcImg.shape, dtype=np.uint8)
    # cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Src")
    cv2.setMouseCallback("Src", OnMouse1)
    cv2.namedWindow("Canvas")
    # cv2.namedWindow("Canvas", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Canvas", OnMouse2)

    SrcPoints = np.float32([[150,123],[439,174],[148,380]])
    CanvasPoints = np.float32([[0,0],[512,0],[0,512]])
    while True:
        Img = SrcImg.copy()
        Canvas = SrcCanvas.copy()
        IPT.drawPoints(Img, np.array(LeftPoint).reshape(-1, 1), (0,0,255), radius=3)
        IPT.drawPoints(Canvas, np.array(RightPoint).reshape(-1, 1), (0,0,255), radius=3)
        for i in SrcPoints:
            IPT.drawPoints(Img, i.reshape(-1, 1), (0,255,0), radius=3)
        for i in CanvasPoints:
            IPT.drawPoints(Canvas, i.reshape(-1, 1), (0,255,0), radius=3)

        cv2.imshow('Src', Img)
        cv2.imshow('Canvas', Canvas)
        Key = chr(cv2.waitKey(30) & 255)
        if Key == 'l':
            if len(SrcPoints) < 3:
                SrcPoints.append(np.array(LeftPoint).reshape(-1))
        elif Key == 'r':
            if len(CanvasPoints) < 3:
                CanvasPoints.append(np.array(RightPoint).reshape(-1))
        elif Key == 'a' or First:
            First = False
            SrcPointsA = np.array(SrcPoints, dtype=np.float32)
            CanvasPointsA = np.array(CanvasPoints, dtype=np.float32)
            print 'SrcPoints:'
            print SrcPointsA
            print 'CanvasPoints:'
            print CanvasPointsA
            AffineMatrix = cv2.getAffineTransform(np.array(SrcPointsA),
                                                  np.array(CanvasPointsA))
            print 'AffineMatrix:\n', AffineMatrix
            AffineImg = cv2.warpAffine(Img, AffineMatrix, (Img.shape[1], Img.shape[0]))
            cv2.imshow('AffineImg', AffineImg)
            cv2.imwrite('../../../Datas/Output/InvAffineLena.png', AffineImg)
        elif Key == 'f':
            SrcPoints = []
            print 'reset SrcPoints'
        elif Key == 'r':
            CanvasPoints =[]
            print 'reset CanvasPoints'
        elif Key == 'q':
            break
