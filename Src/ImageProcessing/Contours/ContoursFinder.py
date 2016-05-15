#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '20/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np

import Src.ToolBox.ImageProcessTool as IPT


if __name__ == '__main__':
    # =========================== Draw Image =========================== #
    # Img = np.zeros((480, 640), dtype=np.uint8)
    # IPT.drawPoints(img=Img, pts_2xn=np.array([[320], [240]]), color=255, radius=30, thickness=1)
    # IPT.drawPoints(img=Img, pts_2xn=np.array([[320], [240]]), color=255, radius=100, thickness=1)
    # IPT.drawPoints(img=Img, pts_2xn=np.array([[320], [240]]), color=255, radius=200, thickness=1)
    # cv2.imwrite(filename='../../../Datas/CirclesInCircle.png', img=Img)

    # Img = np.zeros((480, 640), dtype=np.uint8)
    # cv2.rectangle(Img, (100, 100), (300, 300), color=255, thickness=-1)
    # cv2.imshow('Rect', Img)
    # cv2.waitKey()
    # cv2.imwrite(filename='../../../Datas/Rectangle.png', img=Img)

    Img = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(Img, (100, 200), (300, 400), color=255, thickness=1)
    cv2.imshow('Line', Img)
    cv2.waitKey()
    cv2.imwrite(filename='../../../Datas/Line.png', img=Img)

    # =========================== Find Contours =========================== #
    # SrcImg = cv2.imread('../../../Datas/Numbers.png')
    SrcImg = cv2.imread('../../../Datas/CirclesInCircle.png')
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    # _, BinImg = cv2.threshold(src=GrayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    _, BinImg = cv2.threshold(src=GrayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    # -------------------------- Hierarchy structure -------------------------- #
    Contours, Hierarchy = \
        cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_NONE, offset=None)
    print Hierarchy
    Contours, Hierarchy = \
        cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE, offset=None)
    print Hierarchy
    HierarchyData = Hierarchy[0]
    Start = HierarchyData[0]
    Next = Start[0]
    Prev = Start[1]
    Child = Start[2]
    Parent = Start[3]
    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Bin', BinImg)
    cv2.waitKey()

    # -------------------------- CHAIN_APPROX_NONE -------------------------- #
    SrcImg = cv2.imread('../../../Datas/Rectangle.png')
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    _, BinImg = cv2.threshold(src=GrayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    ApproxNoneImg = SrcImg.copy()
    ApproxSimpleImg = SrcImg.copy()
    Contours, Hierarchy = \
        cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE, offset=None)
    cv2.drawContours(image=ApproxNoneImg, contours=Contours, contourIdx=-1, color=(0, 0, 255))
    print Contours
    Contours, Hierarchy = \
        cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, offset=None)
    print Contours
    cv2.drawContours(image=ApproxSimpleImg, contours=Contours, contourIdx=-1, color=(0, 0, 255))

    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Bin', BinImg)
    cv2.imshow('ApproxNone', ApproxNoneImg)
    cv2.imshow('ApproxSimple', ApproxSimpleImg)
    cv2.waitKey()