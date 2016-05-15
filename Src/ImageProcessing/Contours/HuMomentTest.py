#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '20/04/2015'


import sys
sys.path.append("../../../")

import cv2
import cv2.cv as cv
import math
import numpy as np

import Src.ToolBox.ImageProcessTool as IPT

if __name__ == '__main__':
    Hu7 = np.array([[ 0.63433134],
                    [ 0.18599703],
                    [ 0.1474484 ],
                    [ 0.0292515 ],
                    [ 0.00163499],
                    [ 0.01195863],
                    [-0.00100861]])
    Hu5 = np.array([[  5.71088820e-01],
                    [  1.22423462e-01],
                    [  6.39878035e-03],
                    [  2.59243528e-03],
                    [  9.23185315e-06],
                    [  3.02574364e-04],
                    [ -5.12435864e-06]])
    Hu2 = np.array([[  5.90955715e-01],
                    [  1.35147037e-01],
                    [  1.95177461e-03],
                    [  3.35322845e-04],
                    [  2.14003922e-08],
                    [ -5.75255554e-05],
                    [  2.70429421e-07]])
    Hu2Model = np.array([[  5.68020254e-01],
                         [  1.27258643e-01],
                         [  1.47128840e-03],
                         [  4.74551355e-04],
                         [  3.39563545e-07],
                         [  6.19449311e-05],
                         [  2.04770508e-07]])
    method = 2
    HuMoment1 = Hu2Model
    HuMoment2 = Hu5
    m1 = np.sign(HuMoment1) * np.log(np.abs(HuMoment1))
    m2 = np.sign(HuMoment2) * np.log(np.abs(HuMoment2))
    m3 = np.sign(Hu2) * np.log(np.abs(Hu2))
    print 'm2Model:\n', m1
    print 'm2:\n', m3
    print 'm5:\n', m2
    # Valid1 = np.logical_not(np.isnan(m1))
    # Valid2 = np.logical_not(np.isnan(m2))
    # Valid = np.logical_and(Valid1, Valid2)
    Valid = np.array([True]*7).reshape(HuMoment1.shape)
    if method == 1:
        print np.abs(1/m1[Valid] - 1/m2[Valid])
        print np.abs(1/m1[Valid] - 1/m3[Valid])
        print np.sum(np.abs(1/m1[Valid] - 1/m2[Valid]))
        print np.sum(np.abs(1/m1[Valid] - 1/m3[Valid]))
        DisSimilarity = np.sum(np.abs(1/m1[Valid] - 1/m2[Valid]))
    elif method == 2:
        print np.sum(m1[Valid] - m2[Valid])
        print np.sum(m1[Valid] - m3[Valid])
        print np.sum(np.abs(m1[Valid] - m2[Valid]))
        print np.sum(np.abs(m1[Valid] - m3[Valid]))
        DisSimilarity = np.sum(m1[Valid] - m2[Valid])
    elif method == 3:
        DisSimilarity = np.max(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
    elif method == 4:
        DisSimilarity = np.sum(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
    else:
        raise ValueError, 'method code error.'