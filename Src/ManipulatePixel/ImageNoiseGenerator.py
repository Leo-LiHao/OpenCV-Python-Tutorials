#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '19/01/2015'


import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def addNoise_Gaussian(img, mean, std):
    ImgShape = img.shape
    Noise = np.random.normal(loc=mean, scale=std, size=ImgShape)
    NewImg = img.astype(np.float32) + Noise
    NewImg[NewImg>255] = 255
    NewImg[NewImg<0] = 0
    return NewImg.astype(np.uint8)

def addNoise_SaltAndPepper(img, salt=0.004, pepper=0.004, color_white=False, color_black=False, copyFlag=False):
    ImgShape = img.shape
    if copyFlag:
        NewImg = img.copy()
    else:
        NewImg = img
    # Salt
    SaltNum = np.ceil(img.size * salt)
    SaltCoords = [np.random.randint(low=0, high=i-1, size=(SaltNum)) for i in ImgShape]
    if len(ImgShape) > 2 and color_white:
        NewImg[SaltCoords[0], SaltCoords[1], :] = 255
    else:
        NewImg[SaltCoords] = 255
    # Pepper
    PepperNum = np.ceil(img.size * pepper)
    PepperCoords = [np.random.randint(low=0, high=i-1, size=(PepperNum)) for i in ImgShape]
    if len(ImgShape) > 2 and color_black:
        NewImg[PepperCoords[0], PepperCoords[1], :] = 0
    else:
        NewImg[PepperCoords] = 0
    return NewImg

def addNoise_Poission(img):
    Values = len(np.unique(img))
    Values = 2 ** np.ceil(np.log2(Values))
    NewImg = np.random.poisson(lam=img*Values) / Values
    return NewImg.astype(np.uint8)

def addNoise_Speckle(img):
    ImgShape = img.shape
    Gaussian = np.random.randn(ImgShape[0], ImgShape[1], ImgShape[2])
    NewImg = img + img * Gaussian
    NewImg[NewImg>255] = 255
    NewImg[NewImg<0] = 0
    return NewImg.astype(np.uint8)


if __name__ == '__main__':
    Img = cv2.imread('../../Datas/lena.png')
    # Img = np.ones((500, 500, 3), dtype=np.uint8) * 150
    cv2.imshow('pure', Img)
    NoiseImg_Gaussian = addNoise_Gaussian(img=Img, mean=10, std=10)
    cv2.imshow('noise-gaussian', NoiseImg_Gaussian)

    NoiseImg_SaP = addNoise_SaltAndPepper(img=Img, copyFlag=True)
    cv2.imshow('noise-SaP', NoiseImg_SaP)
    NoiseImg_SaP = addNoise_SaltAndPepper(img=Img, copyFlag=True, color_black=True, color_white=True)
    cv2.imshow('noise-SaP_WhiteAndBlack', NoiseImg_SaP)

    GrayImg = cv2.cvtColor(src=Img, code=cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', GrayImg)
    NoiseGrayImg_SaP = addNoise_SaltAndPepper(img=GrayImg, copyFlag=False)
    cv2.imshow('GrayNoise_SaP', NoiseGrayImg_SaP)

    NoiseGrayImg_Poission = addNoise_Poission(img=Img)
    cv2.imshow('Noise_Poission', NoiseGrayImg_Poission)

    NoiseGrayImg_Speckle = addNoise_Speckle(img=Img)
    cv2.imshow('Noise_Speckle', NoiseGrayImg_Speckle)

    # s = np.random.poisson(1000, 10000)
    # count, bins, ignored = plt.hist(s, 14, normed=True)
    # plt.show()
    cv2.waitKey()
