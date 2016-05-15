#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '21/04/2016'


import cv2
import numpy as np
import sys
sys.path.append('../../')

from Src.ToolBox.CameraCapture      import CameraCapture

import Src.ToolBox.ImageProcessTool     as IPT


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
    Capture = CameraCapture(id=4)
    Capture.open(resolution=(1600, 1200), thread=False)
    # Capture.open(resolution=(1280, 720), thread=False)
    # Capture.open(resolution=(640, 480), thread=False)
    cv2.namedWindow("Src", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Src", OnMouse)

    while True:
        SrcImg = Capture.takePhoto()
        if SrcImg is not None:
            # RotImg, RotMatrix = IPT.rotateImg(srcImg=SrcImg, angle_deg=90)
            # RotImg, RotMatrix = IPT.rotateImg(srcImg=SrcImg, angle_deg=-90)
            # RotGrayImg = cv2.cvtColor(src=RotImg, code=cv2.COLOR_BGR2GRAY)
            RotGrayImg = SrcImg
            IPT.drawRoi(img=RotGrayImg, roi=Roi_xyxy, roiType=IPT.ROI_TYPE_XYXY, color=(0,0,255))
            cv2.imshow("Src", RotGrayImg)
            Key = chr(cv2.waitKey(30) & 0xff)
            if Key in ('q', 'Q'):
                print 'Params: ', Capture.getParams()
                print 'Frame rate: ', Capture.testFrameRate()
                Capture.release()
                break
            elif 'r' == Key:
                print 'roi_xyxy:', Roi_xyxy
                print 'roi_xywh:', IPT.cvtRoi(roi=Roi_xyxy, flag=IPT.ROI_CVT_XYXY2XYWH)