#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/04/2015'


import sys
sys.path.append("../../")

import cv2
import cv2.cv as cv
import numpy as np

from Src.ToolBox.CameraCapture import CameraCapture


class VideoPlayer(object):
    def __init__(self, fileName):
        object.__init__(self)
        self.__Video = cv2.VideoCapture()
        self.__Video.open(fileName)

    @property
    def FrameRate(self):
        return self.__Video.get(cv2.cv.CV_CAP_PROP_FPS)

    @property
    def nowFrameNum(self):
        return self.__Video.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

    @property
    def TotalFrame(self):
        return self.__Video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

    @property
    def nowTime_msec(self):
        return self.__Video.get(cv2.cv.CV_CAP_PROP_POS_MSEC)

    def read(self):
        Flag, Img = self.__Video.read()
        if Flag:
            return Img
        return None

    def release(self):
        self.__Video.release()

    def setFrameNum(self, frameNum):
        return self.__Video.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, frameNum)

    def setTime_msec(self, msec):
        return self.__Video.set(cv2.cv.CV_CAP_PROP_POS_MSEC, msec)




if __name__ == '__main__':
    MyVP = VideoPlayer(fileName='../../Datas/VideoTest.avi')
    print MyVP.FrameRate
    print MyVP.TotalFrame
    while True:
        Img = MyVP.read()
        if Img is not None:
            cv2.imshow('Src', Img)
            Key = chr(cv2.waitKey(int(1000/MyVP.FrameRate)) & 255)
            print MyVP.nowFrameNum
            if MyVP.nowFrameNum == MyVP.TotalFrame:
                break
            print MyVP.nowTime_msec
            if Key == 'q':
                break
            elif Key == 'j':
                MyVP.setFrameNum(100)
            elif Key == 'm':
                MyVP.setTime_msec(1111)
    MyVP.release()
    cv2.destroyAllWindows()


# ----------------------- Video Writter ---------------------- #
# if __name__ == '__main__':
#     Cam = CameraCapture(0)
#     Resolution = (640, 480)
#     Cam.open(resolution=Resolution)
#     print Cam.getParams()
#     Fourcc = cv2.cv.FOURCC('M', 'J', 'P', 'G')
#     FPS = Cam.testFrameRate()
#     FrameSize = Resolution
#     print Fourcc
#     VideoWriter = cv2.VideoWriter(filename='../../Datas/VideoTest.avi',
#                                   fps=FPS,
#                                   fourcc=Fourcc,
#                                   frameSize=FrameSize,
#                                   isColor=True)
#     # VideoWriter = None
#     Record = False
#     while True:
#         Img = Cam.takePhoto()
#         if Img is not None:
#             cv2.imshow('Img', Img)
#             if Record:
#                 VideoWriter.write(Img)
#             Key = chr(cv2.waitKey(15) & 255)
#             if Key == 'q':
#                 break
#             elif Key == 'r':
#                 Record = True
#                 # VideoWriter = cv2.VideoWriter(filename='../../Datas/VideoTest.avi',
#                 #                               fps=30,
#                 #                               fourcc=Fourcc,
#                 #                               frameSize=(640, 480),
#                 #                               isColor=True)
#                 print 'start record'
#             elif Key == 's':
#                 Record = False
#                 print 'stop record'
