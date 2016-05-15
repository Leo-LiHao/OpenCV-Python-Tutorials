#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '21/04/2016'


import cv2
import numpy as np
import sys
sys.path.append('../../../')

from Src.ToolBox.CameraCapture      import CameraCapture

import Src.ToolBox.ImageProcessTool     as IPT
from   ContourAnalyst import ContourAnalyst


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


if __name__ == '__main__':
    # ---------------------- get 0~9 Numbers ---------------------- #
    # global Roi_xyxy
    # cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("Src", OnMouse)
    # while True:
    #     # SrcImg = cv2.imread('../../../Datas/Numbers/OCR.png')
    #     SrcImg = cv2.imread('../../../Datas/Numbers/Numbers.png')
    #     GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    #     _, BinImg = cv2.threshold(src=GrayImg, thresh=100, maxval=255, type=cv2.THRESH_OTSU)
    #
    #     # print 'aaaaa:\n', Roi_xyxy
    #     _, RoiImg = IPT.getRoiImg(BinImg, roi=Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY, copy=True)
    #     IPT.drawRoi(BinImg, Roi_xyxy, IPT.ROI_TYPE_XYXY, 155)
    #     cv2.imshow('Src', BinImg)
    #     cv2.imshow('ROI', RoiImg)
    #     Key = chr(cv2.waitKey(15) & 255)
    #     if Key == 'q':
    #         break
    #     elif '0' <= Key <= '9':
    #         # print 'save: ', cv2.imwrite('../../../Datas/Numbers/'+Key+'.png', RoiImg)
    #         print 'save: ', cv2.imwrite('../../../Datas/Numbers/Numbers_'+Key+'.png', RoiImg)


    # ---------------------- get 0~9 DataBase ---------------------- #
    # cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("Src", OnMouse)
    # cv2.namedWindow("BinImg", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("BinImg", OnMouse)
    # for i in xrange(10):
    #     # SrcImg = cv2.imread('../../../Datas/Numbers/'+str(i)+'.png')
    #     SrcImg = cv2.imread('../../../Datas/Numbers/Numbers_'+str(i)+'.png')
    #     GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    #     _, BinImg = cv2.threshold(src=GrayImg, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    #     Contours, _  = cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    #     # np.savetxt('../../../Datas/Numbers/'+str(i)+'.txt', Contours[0].reshape(-1, 2).T.reshape(2, -1))
    #     np.savetxt('../../../Datas/Numbers/Numbers_'+str(i)+'.txt', Contours[0].reshape(-1, 2).T.reshape(2, -1))
    #     cv2.drawContours(SrcImg, Contours, -1, color=(0, 0, 255))
    #     cv2.imshow('BinImg', BinImg)
    #     cv2.imshow('Src', SrcImg)
    #     cv2.waitKey( )


    # ---------------------- OCR ---------------------- #
    cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    # MatchData = ContourAnalyst.cvtPoints2Contour(np.loadtxt('../../../Datas/Numbers/0.txt'))
    MatchData = ContourAnalyst.cvtPoints2Contour(np.loadtxt('../../../Datas/Numbers/Numbers_1.txt'))
    First = True
    Number = [7, 6, 5, 9, 8, 4, 3, 1, 2, 0]
    ReadImg = '../../../Datas/Numbers/Numbers.png'
    # ReadImg = '../../../Datas/Numbers/OCR.png'
    while True:
        SrcImg = cv2.imread(ReadImg)
        GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
        _, BinImg = cv2.threshold(src=GrayImg, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)

        Contours, _  = cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        Result = None
        MinIdx = None
        for idx, contour in enumerate(Contours):
            MatchResult = ContourAnalyst.matchContour(MatchData, contour, method=4)
            if Result is None:
                Result = MatchResult
                MinIdx = idx
            # print MatchResult
            # cv2.drawContours(SrcImg, [contour], 0, (0,0,255))
            # cv2.imshow('SrcImg', SrcImg)
            # cv2.waitKey()
            if MatchResult < Result:
                Result = MatchResult
                MinIdx = idx
            if First:
                print Number[idx], ': ', MatchResult
            #     # print 'contour Moment: ', cv2.moments(contour)
            #     # print 'contour Center: ', ContourAnalyst.getCentroid(contour)
            #     print 'HuMoment: ', cv2.HuMoments(m=cv2.moments(contour))
            if abs(MatchResult) < 0.1:
                cv2.drawContours(SrcImg, [contour], 0, (0, 255, 0), thickness=10)

        cv2.drawContours(SrcImg, [Contours[MinIdx]], 0, (0, 0, 255), thickness=5)

        if First:
            First = False
        # raw_input()
        cv2.imshow('Src', SrcImg)
        cv2.imshow('Bin', BinImg)
        Key = chr(cv2.waitKey(15) & 255)
        if Key == 'q':
            break
        elif '0' <= Key <= '9':
            print '-------------------------------------------'
            print 'read: ', Key
            print '-------------------------------------------'
            First = True
            # MatchData = ContourAnalyst.cvtPoints2Contour(np.loadtxt('../../../Datas/Numbers/'+Key+'.txt'))
            MatchData = ContourAnalyst.cvtPoints2Contour(np.loadtxt('../../../Datas/Numbers/Numbers_'+Key+'.txt'))
            # Offset = np.array([[30],
            #                    [40]]).reshape(-1, 1, 2)
            # MatchData = (MatchData + Offset) * 5
            # print 'matchData: ', cv2.moments(MatchData)
            # print 'Center: ', ContourAnalyst.getCentroid(MatchData)
            print 'HuMoment: ', cv2.HuMoments(m=cv2.moments(MatchData))
            # ReadImg = '../../../Datas/Numbers/Numbers_'+ Key +'.png'
            Canvas = np.zeros((300, 300), dtype=np.uint8)
            cv2.drawContours(Canvas, [MatchData], 0, 155)
            cv2.imshow('Canvas', Canvas)
            # cv2.waitKey()