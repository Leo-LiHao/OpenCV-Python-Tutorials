#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '18/04/2015'


import sys
sys.path.append("../../../")

import cv2
import numpy as np
import GaussianLaplacianPyramid as Pyramid


if __name__ == '__main__':
    AppleImg = cv2.imread('../../../Datas/Apple.png')
    OrangeImg = cv2.imread('../../../Datas/Orange.png')

    AppleImg = cv2.resize(src=AppleImg, dsize=(OrangeImg.shape[1], OrangeImg.shape[0]))
    AppleGauPyramid = Pyramid.getGaussianPyramid(AppleImg, 6)
    OrangeGauPyramid = Pyramid.getGaussianPyramid(OrangeImg, 6)

    AppleLapPyramid = Pyramid.getLaplacianPyramid(AppleImg, 5)
    OrangeLapPyramid = Pyramid.getLaplacianPyramid(OrangeImg, 5)

    MixImgPyr = []
    for AppleLap, OrangeLap in zip(AppleLapPyramid, OrangeLapPyramid):
        h, w, d = AppleLap.shape
        MixImg = np.hstack((AppleLap[:, :w/2], OrangeLap[:, w/2:]))
        MixImgPyr.append(MixImg)

    h, w, d = AppleGauPyramid[-1].shape
    BlendImg = np.hstack((AppleGauPyramid[-1][:, :w/2], OrangeGauPyramid[-1][:, w/2:]))
    for i in xrange(len(MixImgPyr)-1, -1, -1):
        BlendImg = cv2.pyrUp(BlendImg)
        BlendImg = cv2.add(BlendImg, MixImgPyr[i])
        cv2.namedWindow('BlendImg'+str(i), cv2.WINDOW_NORMAL)
        cv2.imshow('BlendImg'+str(i), BlendImg)

    cv2.imshow('Apple', AppleImg)
    cv2.imshow('OrangeImg', OrangeImg)
    cv2.imshow('BlendImg', BlendImg)
    cv2.waitKey()

    # # generate Gaussian pyramid for A
    # G = AppleImg.copy()
    # gpA = [G]
    # for i in xrange(6):
    #     G = cv2.pyrDown(G)
    #     gpA.append(G)
    #
    # # generate Gaussian pyramid for B
    # G = OrangeImg.copy()
    # gpB = [G]
    # for i in xrange(6):
    #     G = cv2.pyrDown(G)
    #     gpB.append(G)
    #
    # # generate Laplacian Pyramid for A
    # lpA = [gpA[5]]
    # for i in xrange(5,0,-1):
    #     GE = cv2.pyrUp(gpA[i])
    #     L = cv2.subtract(gpA[i-1],GE)
    #     lpA.append(L)
    #
    # # generate Laplacian Pyramid for B
    # lpB = [gpB[5]]
    # for i in xrange(5,0,-1):
    #     GE = cv2.pyrUp(gpB[i])
    #     L = cv2.subtract(gpB[i-1],GE)
    #     lpB.append(L)
    #
    # # Now add left and right halves of images in each level
    # LS = []
    # for la,lb in zip(lpA,lpB):
    #     rows,cols,dpt = la.shape
    #     ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
    #     LS.append(ls)
    #
    # # now reconstruct
    # ls_ = LS[0]
    # for i in xrange(1,6):
    #     ls_ = cv2.pyrUp(ls_)
    #     ls_ = cv2.add(ls_, LS[i])
    #
    #
    # cv2.imshow('aaa', ls_)
    # print np.equal(ls_, BlendImg).all()
    # cv2.waitKey()




