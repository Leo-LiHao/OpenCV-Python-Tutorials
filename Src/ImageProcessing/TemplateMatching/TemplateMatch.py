#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '23/04/2015'


import sys
sys.path.append("../../../")

import cv2
import time
import numpy as np

import Src.ToolBox.ImageProcessTool     as IPT


if __name__ == '__main__':
    # # -------------------------- test all method  --------------------------- #
    # SrcImgPath = '../../../Datas/Numbers/Numbers.png'
    # SrcImg = cv2.imread(SrcImgPath)
    # GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)
    # Methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
    #            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    #
    # Number = '0'
    # while True:
    #     TemplatePath = '../../../Datas/Numbers/Numbers_' + Number + '.png'
    #     TemplateImg = cv2.imread(TemplatePath, 0)
    #     Height, Weight = TemplateImg.shape
    #     for idx, methodStr in enumerate(Methods):
    #         Method = eval(methodStr)
    #         T0 = time.time()
    #         MatchResult = cv2.matchTemplate(image=GrayImg, templ=TemplateImg, method=Method)
    #         print methodStr, 'use time:', time.time() - T0
    #         DrawImg = SrcImg.copy()
    #         MinVal, MaxVal, MinLoc, MaxLoc = cv2.minMaxLoc(src=MatchResult)
    #         if Method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
    #             Roi_xywh = [MinLoc[0], MinLoc[1], Weight, Height]
    #             print 'MinVal:', MinVal
    #         else:
    #             Roi_xywh = [MaxLoc[0], MaxLoc[1], Weight, Height]
    #             print 'MaxVal:', MaxVal
    #         IPT.drawRoi(img=DrawImg, roi=Roi_xywh, roiType=IPT.ROI_TYPE_XYWH, color=(0, 0, 255))
    #         cv2.imshow(methodStr, DrawImg)
    #
    #     Key = chr(cv2.waitKey() & 255)
    #     if Key == 'q':
    #         break
    #     elif '0' <= Key <= '9':
    #         print 'Key: ', Key
    #         Number = Key

    # -------------------------- match all --------------------------- #
    SrcImgPath = '../../../Datas/Numbers/OCR.png'
    SrcImg = cv2.imread(SrcImgPath)
    GrayImg = cv2.cvtColor(SrcImg, cv2.COLOR_BGR2GRAY)

    Number = '0'
    while True:
        TemplatePath = '../../../Datas/Numbers/' + Number + '.png'
        TemplateImg = cv2.imread(TemplatePath, 0)
        Height, Weight = TemplateImg.shape
        MatchResult = cv2.matchTemplate(image=GrayImg, templ=TemplateImg, method=cv2.TM_CCOEFF_NORMED)

        DrawImg = SrcImg.copy()
        Thresh = 0.8
        Loc = np.where(MatchResult >= Thresh)
        for pt in zip(*Loc[:: -1]):
            Roi_xywh = [pt[0], pt[1], Weight, Height]
            IPT.drawRoi(img=DrawImg, roi=Roi_xywh, roiType=IPT.ROI_TYPE_XYWH, color=(0, 0, 255))
        cv2.imshow('MatchResult', MatchResult)
        cv2.imshow('Result', DrawImg)

        Key = chr(cv2.waitKey() & 255)
        if Key == 'q':
            break
        elif '0' <= Key <= '9':
            print 'Key: ', Key
            Number = Key