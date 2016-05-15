#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/04/2015'


import sys
sys.path.append("../../../")

import cv2
import cv2.cv as cv
import numpy as np

from Src.ToolBox.CameraCapture import CameraCapture


if __name__ == '__main__':
    Cam = CameraCapture(0)
    Resolution = (640, 480)
    Cam.open(resolution=Resolution)
    while True:
        Img = Cam.takePhoto()

        if Img is not None:
            HSV = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
            import time
            T0 = time.time()
            H, S, V = cv2.split(HSV)
            print 'split use time:', time.time() - T0
            T0 = time.time()
            H = HSV[:,:,0]
            S = HSV[:,:,1]
            V = HSV[:,:,2]
            print 'idx use time:', time.time() - T0
            LowerBlue = np.array([100, 100, 50])
            UpperBlue = np.array([130, 255, 255])
            mask = cv2.inRange(HSV, LowerBlue, UpperBlue)
            BlueThings = cv2.bitwise_and(Img, Img, mask=mask)
            cv2.imshow('Mask', mask)
            cv2.imshow('Img', Img)
            cv2.imshow('H', H)
            cv2.imshow('Blue', BlueThings)
            # cv2.imshow('S', S)
            # cv2.imshow('V', V)
            Key = chr(cv2.waitKey(15) & 255)
            if Key == 'q':
                break
            elif Key == 's':
                print 'save img:', cv2.imwrite(filename='../../../Datas/Output/ColorSpace_src.png', img=Img)
                print 'save Blue:', cv2.imwrite(filename='../../../Datas/Output/ColorSpace_BlueThings.png', img=BlueThings)
    Cam.release()
    cv2.destroyAllWindows()
