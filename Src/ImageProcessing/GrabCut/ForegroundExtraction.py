#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '26/04/2016'


import sys
sys.path.append('../../../')

import cv2
import numpy as np

import Src.ToolBox.ImageProcessTool     as IPT


MASK_THICKNESS = 3
BLUE  = (255,   0,   0)     # rectangle color
RED   = (  0,   0, 255)     # PR BG
GREEN = (  0, 255,   0)     # PR FG
BLACK = (  0,   0,   0)     # sure BG
WHITE = (255, 255, 255)     # sure FG

DrawColors = {cv2.GC_BGD:    BLACK,
              cv2.GC_FGD:    WHITE,
              cv2.GC_PR_BGD: RED,
              cv2.GC_PR_FGD: BLUE}
MaskType = cv2.GC_BGD

# FilePath = '../../../Datas/Coins.jpg'
FilePath = '../../../Datas/lena.png'
SrcImg = cv2.imread(FilePath)
DrawMaskImg = SrcImg.copy()
ShowAllImg = DrawMaskImg.copy()
MaskImg = np.ones(SrcImg.shape[:2], dtype=np.uint8) * cv2.GC_PR_BGD
DownPoint = [0,0]
Roi_xyxy = np.array([0,0,0,0])
RightButtonOn = False
LeftButtonOn = False


def OnMouse(event, x, y, flags, param):
    global DrawMaskImg, ShowAllImg, MaskImg
    global DownPoint, RightButtonOn, LeftButtonOn
    global Roi_xyxy
    global MaskType, DrawColors

    if cv2.EVENT_RBUTTONDOWN == event:
        RightButtonOn = True
        DownPoint = [x, y]
    elif cv2.EVENT_MOUSEMOVE == event:
        if RightButtonOn:
            Roi_xyxy[0:2] = [min(DownPoint[0], x), min(DownPoint[1], y)]
            Roi_xyxy[2:4] = [max(DownPoint[0], x), max(DownPoint[1], y)]
    elif cv2.EVENT_RBUTTONUP == event:
        RightButtonOn = False
        if DownPoint[0] != x and DownPoint[1] != y:
            Roi_xyxy[0:2] = [min(DownPoint[0], x), min(DownPoint[1], y)]
            Roi_xyxy[2:4] = [max(DownPoint[0], x), max(DownPoint[1], y)]

    if cv2.EVENT_LBUTTONDOWN == event:
        LeftButtonOn = True
        cv2.circle(img=DrawMaskImg, center=(x, y), radius=MASK_THICKNESS, color=DrawColors[MaskType], thickness=-1)
        cv2.circle(img=MaskImg, center=(x, y), radius=MASK_THICKNESS, color=int(MaskType), thickness=-1)
    elif cv2.EVENT_MOUSEMOVE == event:
        if LeftButtonOn:
            cv2.circle(img=DrawMaskImg, center=(x, y), radius=MASK_THICKNESS, color=DrawColors[MaskType], thickness=-1)
            cv2.circle(img=MaskImg, center=(x, y), radius=MASK_THICKNESS, color=int(MaskType), thickness=-1)
    elif cv2.EVENT_LBUTTONUP == event:
        LeftButtonOn = False
        cv2.circle(img=DrawMaskImg, center=(x, y), radius=MASK_THICKNESS, color=DrawColors[MaskType], thickness=-1)
        cv2.circle(img=MaskImg, center=(x, y), radius=MASK_THICKNESS, color=int(MaskType), thickness=-1)


def drawGrabCutMask(canvas, mask, colors):
    for key in colors.keys():
        canvas[mask==key] = colors[key]

def getGrabCutResult(srcImg, mask, flag):
    if 0 == flag:
        InterestedMask = np.where((mask==cv2.GC_BGD) + (mask==cv2.GC_PR_BGD), 255, 0).astype('uint8')
    elif 1 == flag:
        InterestedMask = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
    GrabCutImg = cv2.bitwise_and(srcImg, srcImg, mask=InterestedMask)
    return GrabCutImg

if __name__ == '__main__':
    ShowImg = SrcImg.copy()
    GrabCutMaskImg = np.zeros(SrcImg.shape, dtype=np.uint8)
    cv2.namedWindow('Src')
    cv2.setMouseCallback('Src', OnMouse)
    while True:
        ShowAllImg = DrawMaskImg.copy()
        if not np.allclose(Roi_xyxy, [0, 0, 0, 0]):
            IPT.drawRoi(img=ShowAllImg, roi=Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY, color=(0, 255, 0))

        cv2.imshow('Src', ShowAllImg)
        Key = chr(cv2.waitKey(15) & 255)
        if Key == 'q':
            break
        elif Key == '0':
            print " mark background regions with left mouse button \n"
            MaskType = cv2.GC_BGD
        elif Key == '1':
            print " mark foreground regions with left mouse button \n"
            MaskType = cv2.GC_FGD
        elif Key == '2':
            print " mark possible background regions with left mouse button \n"
            MaskType = cv2.GC_PR_BGD
        elif Key == '3':
            print " mark possible foreground regions with left mouse button \n"
            MaskType = cv2.GC_PR_FGD
        elif Key == 'c':
            print 'use rect to grab cut'
            Roi_xywh = IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            MaskResult = np.zeros(SrcImg.shape[:2], dtype=np.uint8)
            cv2.grabCut(img=SrcImg, mask=MaskResult, rect=tuple(Roi_xywh),
                        bgdModel=bgdModel, fgdModel=fgdModel,
                        iterCount=1, mode=cv2.GC_INIT_WITH_RECT)
            drawGrabCutMask(canvas=GrabCutMaskImg, mask=MaskResult, colors=DrawColors)
            GrabCutImg = getGrabCutResult(srcImg=SrcImg, mask=MaskResult, flag=1)
            cv2.imshow('Result_rect', GrabCutImg)
            cv2.imshow('Mask_rect', GrabCutMaskImg)
        elif Key == 'm':
            print 'use mask to grab cut'
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            MaskResult = MaskImg.copy()
            cv2.grabCut(img=SrcImg, mask=MaskResult, rect=None,
                        bgdModel=bgdModel, fgdModel=fgdModel,
                        iterCount=1, mode=cv2.GC_INIT_WITH_MASK)
            drawGrabCutMask(canvas=GrabCutMaskImg, mask=MaskResult, colors=DrawColors)
            GrabCutImg = getGrabCutResult(srcImg=SrcImg, mask=MaskResult, flag=1)
            cv2.imshow('Result_mask', GrabCutImg)
            cv2.imshow('Mask_mask', GrabCutMaskImg)
        elif Key == 'b':
            print 'use rect&mask to grab cut'
            Roi_xywh = IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            MaskResult = MaskImg.copy()
            cv2.grabCut(img=SrcImg, mask=MaskResult, rect=tuple(Roi_xywh),
                        bgdModel=bgdModel, fgdModel=fgdModel,
                        iterCount=3, mode=cv2.GC_INIT_WITH_MASK | cv2.GC_INIT_WITH_RECT)
            drawGrabCutMask(canvas=GrabCutMaskImg, mask=MaskResult, colors=DrawColors)
            GrabCutImg = getGrabCutResult(srcImg=SrcImg, mask=MaskResult, flag=1)
            cv2.imshow('Result_both', GrabCutImg)
            cv2.imshow('Mask_both', GrabCutMaskImg)
        elif Key == 'l':
            print 'use rect&mask to grab cut'
            Roi_xywh = IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            MaskResult = MaskImg.copy()
            cv2.grabCut(img=SrcImg, mask=MaskResult, rect=tuple(Roi_xywh),
                        bgdModel=bgdModel, fgdModel=fgdModel,
                        iterCount=3, mode=cv2.GC_INIT_WITH_MASK)
            drawGrabCutMask(canvas=GrabCutMaskImg, mask=MaskResult, colors=DrawColors)
            GrabCutImg = getGrabCutResult(srcImg=SrcImg, mask=MaskResult, flag=1)
            cv2.imshow('Result_bothL', GrabCutImg)
            cv2.imshow('Mask_bothL', GrabCutMaskImg)
