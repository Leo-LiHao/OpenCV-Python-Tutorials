#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/04/2015'


import cv2
import numpy as np


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/InsertMachine/SrcA.bmp', cv2.CV_LOAD_IMAGE_COLOR)
    B, G, R = cv2.split(Img)
    cv2.imshow('B', B)
    cv2.imshow('G', G)
    cv2.imshow('Img', np.dstack((B, G, R)))
    cv2.waitKey()