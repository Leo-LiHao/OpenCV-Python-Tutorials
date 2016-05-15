#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/04/2015'


import sys
sys.path.append("../../")

import cv2
import numpy as np


def callBack(x):
    print 'Oh!', x


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/InsertMachine/SrcA.bmp', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imshow('Img', Img)
    cv2.namedWindow('BinImg', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('thresh', 'BinImg', 0, 255, callBack)
    while True:
        Thresh = cv2.getTrackbarPos('thresh', 'BinImg')
        _, BinImg = cv2.threshold(src=Img, thresh=Thresh, maxval=255, type=cv2.THRESH_BINARY_INV)
        cv2.imshow('BinImg', BinImg)
        Key = chr(cv2.waitKey(12) & 255)
        if Key == 'q':
            break
    cv2.destroyAllWindows()