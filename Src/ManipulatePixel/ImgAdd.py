#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '27/01/2015'

import sys
sys.path.append("../../")

import cv2
import numpy as np

import Src.ToolBox.ImageProcessTool as IPT


def addLogo(src, logoImg, pos=(0,0)):
    LogoShape = logoImg.shape
    Height, Width = LogoShape[:2]
    Roi_xywh = [pos[0], pos[1], Width, Height]
    _, RoiImg = IPT.getRoiImg(img=src, roi=Roi_xywh, roiType=IPT.ROI_TYPE_XYWH, copy=False)
    RoiImg[:] = cv2.addWeighted(RoiImg, 0.5, logoImg, 0.5, 0)


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/lena.png')

    Logo = cv2.imread('../../Datas/Logo.jpg')
    LogoImg = cv2.resize(src=Logo, dsize=(Img.shape[1]/10, Img.shape[0]/10))
    BackImg = cv2.resize(src=Logo, dsize=(Img.shape[1], Img.shape[0]))

    AddBackImg = cv2.addWeighted(Img, 0.5, BackImg, 0.5, 0)
    addLogo(Img, Logo)
    cv2.imshow('Back', AddBackImg)
    cv2.imshow('Logo', Img)
    cv2.waitKey()

    B, G, R = cv2.split(Img)
    print 'split is equal to Img[:,:,0~2]'
    print 'B, G, R = cv2.split(Img)'
    print 'B is equal Img[:,:,0] ? ', np.allclose(B, Img[:,:,0])
    print 'G is equal Img[:,:,1] ? ', np.allclose(G, Img[:,:,1])
    print 'R is equal Img[:,:,2] ? ', np.allclose(R, Img[:,:,2])


