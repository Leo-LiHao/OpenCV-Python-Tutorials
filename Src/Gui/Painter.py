#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '03/04/2015'


import sys
sys.path.append("../../")

import cv2
import numpy as np


if __name__ == '__main__':
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = np.array([[10, 10], [10, 50], [50, 50], [50, 10]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,255))
    cv2.imshow('img', img)
    cv2.waitKey()