#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '17/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np


if __name__ == '__main__':
    Img = cv2.imread('../../../Datas/InsertMachine/SrcA.bmp')
    Img = cv2.resize(Img, (Img.shape[1]/2, Img.shape[0]/2))
    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)


    Kernel = np.ones((9,9), dtype=np.uint8)
    IterNum = 1
    ErodeImg = cv2.erode(src=GrayImg, kernel=Kernel, iterations=IterNum)
    DilateImg = cv2.dilate(src=GrayImg, kernel=Kernel, iterations=IterNum)

    # cv2.imshow('Src', Img)
    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Erode', ErodeImg)
    cv2.imshow('Dilate', DilateImg)

    GradientImg = cv2.morphologyEx(src=GrayImg, op=cv2.MORPH_GRADIENT, kernel=Kernel, iterations=IterNum)
    BasicGradient = DilateImg - ErodeImg
    cv2.imshow('MorGradient', GradientImg)
    cv2.imshow('BasicGradient', BasicGradient)
    print 'MorGradient == BasicGradient?', np.equal(GradientImg, BasicGradient).all()
    cv2.waitKey()

    # ---------------------------- MorOpen & Top hat---------------------------- #
    Kernel = np.ones((29, 29), dtype=np.uint8)
    IterNum = 1
    OpenImg = cv2.morphologyEx(src=GrayImg, op=cv2.MORPH_OPEN, kernel=Kernel, iterations=IterNum)
    ErodeImg = cv2.erode(src=GrayImg, kernel=Kernel, iterations=IterNum)
    DilateImg = cv2.dilate(src=ErodeImg, kernel=Kernel, iterations=IterNum)
    cv2.destroyAllWindows()
    print 'src == MorOpen?', np.equal(GrayImg, OpenImg).all()
    print 'BasicOpen == MorOpen', np.equal(DilateImg, OpenImg).all()

    TopHatImg = cv2.morphologyEx(src=GrayImg, op=cv2.MORPH_TOPHAT, kernel=Kernel, iterations=IterNum)
    BasicTop = GrayImg - OpenImg
    print 'TopHatImg == BasicTop?', np.equal(TopHatImg, BasicTop).all()
    cv2.imshow('Src', GrayImg)
    cv2.imshow('MorOpen', OpenImg)
    cv2.imshow('BasicOpen', DilateImg)
    cv2.imshow('TopHat', TopHatImg)
    cv2.imshow('BasicTop', BasicTop)
    cv2.waitKey()

    # ---------------------------- MorClose & black hat---------------------------- #
    Kernel = np.ones((29, 29), dtype=np.uint8)
    IterNum = 1
    CloseImg = cv2.morphologyEx(src=GrayImg, op=cv2.MORPH_CLOSE, kernel=Kernel, iterations=IterNum)
    DilateImg = cv2.dilate(src=GrayImg, kernel=Kernel, iterations=IterNum)
    ErodeImg = cv2.erode(src=DilateImg, kernel=Kernel, iterations=IterNum)
    cv2.destroyAllWindows()
    print 'src == MorClose?', np.equal(GrayImg, CloseImg).all()
    print 'BasicClose == MorClose?', np.equal(ErodeImg, CloseImg).all()

    BlackHatImg = cv2.morphologyEx(src=GrayImg, op=cv2.MORPH_BLACKHAT, kernel=Kernel, iterations=IterNum)
    BasicBlack = CloseImg - GrayImg
    print 'BlackHatImg == BasicBlack?', np.equal(BlackHatImg, BasicBlack).all()
    cv2.imshow('Src', GrayImg)
    cv2.imshow('MorClose', CloseImg)
    cv2.imshow('BasicClose', ErodeImg)
    cv2.imshow('BlackHat', BlackHatImg)
    cv2.imshow('BasicBlack', BasicBlack)
    cv2.waitKey()