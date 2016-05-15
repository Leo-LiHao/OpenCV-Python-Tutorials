#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '04/04/2015'


import sys
sys.path.append("../../")

import cv2
import numpy as np


def callBack(x):
    print 'Oh!', x


if __name__ == '__main__':
    # Img = np.array([[1, 8],
    #                 [2, 7]], dtype=np.uint8)
    # # mask = np.array([[0, 1],
    # #                  [1, 0]], dtype=np.uint8)
    # mask = np.array([[1]], dtype=np.uint8)
    # print cv2.bitwise_and(Img, Img, mask=mask)

    Img1 = cv2.imread('../../Datas/lena.png', cv2.CV_LOAD_IMAGE_COLOR)
    Img2 = cv2.imread('../../Datas/Logo.jpg', cv2.CV_LOAD_IMAGE_COLOR)
    Img2 = cv2.resize(Img2, (Img1.shape[1], Img1.shape[0]))

    cv2.imshow('1', Img1)
    cv2.imshow('2', Img2)
    cv2.namedWindow('Img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('merge', 'Img', 0, 100, callBack)
    while True:
        MergeValue = cv2.getTrackbarPos('merge', 'Img')
        MergePercent = MergeValue / 100.0
        Img = cv2.addWeighted(Img1, MergePercent, Img2, 1-MergePercent, 0)
        cv2.imshow('Img', Img)
        Key = chr(cv2.waitKey(12) & 255)
        if Key == 'q':
            break
    cv2.destroyAllWindows()