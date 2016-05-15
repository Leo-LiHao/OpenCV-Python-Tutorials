#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '02/05/2016'


import sys
sys.path.append('../../../')

import cv2
import time
import numpy as np

from Src.ImageProcessing.Contours.ContourAnalyst import ContourAnalyst
import Src.ToolBox.ImageProcessTool     as IPT
import BruteForceMatching

RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)


if __name__ == '__main__':
    ModelImgPath = '../../../Datas/box.png'
    QueryImgPath = '../../../Datas/box_in_scene.png'
    ModelImg = cv2.imread(ModelImgPath)
    QueryImg = cv2.imread(QueryImgPath)
    ModelGrayImg = cv2.cvtColor(src=ModelImg, code=cv2.COLOR_BGR2GRAY)
    QueryGrayImg = cv2.cvtColor(src=QueryImg, code=cv2.COLOR_BGR2GRAY)

    # -------------- Detector --------------- #
    ORB = cv2.ORB()
    SIFT = cv2.SIFT()

    # -------------- detect features --------------- #
    ModelKeyPoints_ORB, ModelDescriptions_ORB = ORB.detectAndCompute(ModelGrayImg, mask=None)
    QueryKeyPoints_ORB, QueryDescriptions_ORB = ORB.detectAndCompute(QueryGrayImg, mask=None)
    ModelShowImg_ORB = cv2.drawKeypoints(image=ModelImg, keypoints=ModelKeyPoints_ORB, color=RED)
    QueryShowImg_ORB = cv2.drawKeypoints(image=QueryImg, keypoints=QueryKeyPoints_ORB, color=RED)
    cv2.imshow('ModelShowImg_ORB', ModelShowImg_ORB)
    cv2.imshow('QueryShowImg_ORB', QueryShowImg_ORB)
    
    ModelKeyPoints_SIFT, ModelDescriptions_SIFT = SIFT.detectAndCompute(ModelGrayImg, mask=None)
    QueryKeyPoints_SIFT, QueryDescriptions_SIFT = SIFT.detectAndCompute(QueryGrayImg, mask=None)
    ModelShowImg_SIFT = cv2.drawKeypoints(image=ModelImg, keypoints=ModelKeyPoints_SIFT, color=RED)
    QueryShowImg_SIFT = cv2.drawKeypoints(image=QueryImg, keypoints=QueryKeyPoints_SIFT, color=RED)
    cv2.imshow('ModelShowImg_SIFT', ModelShowImg_SIFT)
    cv2.imshow('QueryShowImg_SIFT', QueryShowImg_SIFT)
    cv2.waitKey()

    # -------------- matching --------------- #
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH    = 6
    FLANNParams_ORB = dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
    FLANNParams_SIFT = dict(algorithm = FLANN_INDEX_KDTREE, tree=5)
    SearchParams = dict(checks=50)  # or pass empty dictionary
    FLANN_ORB = cv2.FlannBasedMatcher(FLANNParams_ORB, SearchParams)
    FLANN_SIFT = cv2.FlannBasedMatcher(FLANNParams_SIFT, SearchParams)

    MatchResult_ORB = FLANN_ORB.knnMatch(ModelDescriptions_ORB, QueryDescriptions_ORB, k=2)
    MatchResult_SIFT = FLANN_SIFT.knnMatch(ModelDescriptions_SIFT, QueryDescriptions_SIFT, k=2)
    # Apply ratio test
    GoodMatches_ORB = []
    for m, n in MatchResult_ORB:
        if m.distance < 0.75*n.distance:
            GoodMatches_ORB.append(m)
    print 'GoodMatches_ORB: ', len(GoodMatches_ORB)
    GoodMatches_SIFT = []
    for m, n in MatchResult_SIFT:
        if m.distance < 0.75*n.distance:
            GoodMatches_SIFT.append(m)
    print 'GoodMatches_SIFT: ', len(GoodMatches_SIFT)

    MatchShowImg_SIFT = \
        BruteForceMatching.drawMatches(img1=ModelImg, keyPoints1=ModelKeyPoints_SIFT,
                                       img2=QueryImg, keyPoints2=QueryKeyPoints_SIFT, matches=GoodMatches_SIFT)
    MatchShowImg_ORB = \
        BruteForceMatching.drawMatches(img1=ModelImg, keyPoints1=ModelKeyPoints_ORB,
                                       img2=QueryImg, keyPoints2=QueryKeyPoints_ORB, matches=GoodMatches_ORB)

    cv2.imshow('MatchShowImg_SIFT', MatchShowImg_SIFT)
    cv2.imshow('MatchShowImg_ORB', MatchShowImg_ORB)
    cv2.waitKey()
    cv2.destroyAllWindows()