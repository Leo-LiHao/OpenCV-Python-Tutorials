#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '16/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np


if __name__ == '__main__':
    Img = cv2.imread('../../../Datas/j.png')

    Kernel = (7, 7)
    print 'Kernel: ',
    print 'MORPH_RECT:\n', cv2.getStructuringElement(cv2.MORPH_RECT, Kernel)
    print 'MORPH_ELLIPSE:\n', cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Kernel)
    print 'MORPH_CROSS:\n', cv2.getStructuringElement(cv2.MORPH_CROSS, Kernel)