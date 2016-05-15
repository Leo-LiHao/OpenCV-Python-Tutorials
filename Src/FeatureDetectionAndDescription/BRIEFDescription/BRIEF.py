#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '30/04/2016'


import sys
sys.path.append('../../../')

import cv2


if __name__ == '__main__':
    FilePath = '../../../Datas/Circleboard.png'
    SrcImg = cv2.imread(FilePath)
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)

    STAR = cv2.FeatureDetector_create("STAR")
    BRIEF = cv2.DescriptorExtractor_create("BRIEF")
    KeyPoints = STAR.detect(GrayImg)
    KeyPoints, Descriptions = BRIEF.compute(GrayImg, KeyPoints)

    print 'STAR detector + BRIEF descriptor'
    print BRIEF.getInt('bytes')
    print Descriptions.shape

    SIFT = cv2.FeatureDetector_create("SIFT")
    BRIEF = cv2.DescriptorExtractor_create("BRIEF")
    SIFTDescriptor = cv2.DescriptorExtractor_create("SIFT")
    KeyPoints_SIFT = SIFT.detect(GrayImg)
    KeyPoints_SIFT, Descriptions_BRIEF = BRIEF.compute(GrayImg, KeyPoints_SIFT)
    KeyPoints_SIFT, Descriptions_SIFT = SIFTDescriptor.compute(GrayImg, KeyPoints_SIFT)

    print 'SIFT detector + BRIEF & SIFT descriptor'
    print 'BRIEF descriptor'
    print 'dtype: ', Descriptions_BRIEF.dtype
    print 'shape: ', Descriptions_BRIEF.shape
    print 'SIFT descriptor'
    print 'dtype: ', Descriptions_SIFT.dtype
    print 'shape: ', Descriptions_SIFT.shape

