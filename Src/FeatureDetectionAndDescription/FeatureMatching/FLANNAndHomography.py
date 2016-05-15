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


MIN_MATCH_COUNT = 3

RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


if __name__ == '__main__':
    ModelImgPath = '../../../Datas/box.png'
    QueryImgPath = '../../../Datas/box_in_scene.png'
    ModelImg = cv2.imread(ModelImgPath)
    QueryImg = cv2.imread(QueryImgPath)
    ModelGrayImg = cv2.cvtColor(src=ModelImg, code=cv2.COLOR_BGR2GRAY)
    QueryGrayImg = cv2.cvtColor(src=QueryImg, code=cv2.COLOR_BGR2GRAY)

    # -------------- Detector --------------- #
    SIFT = cv2.SIFT()

    # -------------- detect features --------------- #
    ModelKeyPoints_SIFT, ModelDescriptions_SIFT = SIFT.detectAndCompute(ModelGrayImg, mask=None)
    QueryKeyPoints_SIFT, QueryDescriptions_SIFT = SIFT.detectAndCompute(QueryGrayImg, mask=None)
    ModelShowImg_SIFT = cv2.drawKeypoints(image=ModelImg, keypoints=ModelKeyPoints_SIFT, color=RED)
    QueryShowImg_SIFT = cv2.drawKeypoints(image=QueryImg, keypoints=QueryKeyPoints_SIFT, color=RED)
    cv2.imshow('ModelShowImg_SIFT', ModelShowImg_SIFT)
    cv2.imshow('QueryShowImg_SIFT', QueryShowImg_SIFT)

    # -------------- matching --------------- #
    FLANN_INDEX_KDTREE = 1
    FLANN_INDEX_LSH    = 6
    FLANNParams_ORB = dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
    FLANNParams_SIFT = dict(algorithm = FLANN_INDEX_KDTREE, tree=5)
    SearchParams = dict(checks=50)  # or pass empty dictionary
    FLANN_SIFT = cv2.FlannBasedMatcher(FLANNParams_SIFT, SearchParams)
    MatchResult_SIFT = FLANN_SIFT.knnMatch(ModelDescriptions_SIFT, QueryDescriptions_SIFT, k=2)
    # Apply ratio test
    GoodMatches_SIFT = []
    for m, n in MatchResult_SIFT:
        if m.distance < 0.75*n.distance:
            GoodMatches_SIFT.append(m)
    print 'GoodMatches_SIFT: ', len(GoodMatches_SIFT)

    if len(GoodMatches_SIFT) > MIN_MATCH_COUNT:
        ModelPts_nx1x2 = np.float32([ModelKeyPoints_SIFT[m.queryIdx].pt for m in GoodMatches_SIFT]).reshape(-1,1,2)
        QueryPts_nx1x2 = np.float32([QueryKeyPoints_SIFT[m.trainIdx].pt for m in GoodMatches_SIFT]).reshape(-1,1,2)

        HomoMatrix, Mask = cv2.findHomography(ModelPts_nx1x2, QueryPts_nx1x2, cv2.RANSAC, 5.0)
        MatchesMask = Mask.ravel().tolist()
        h, w = ModelImg.shape[:2]
        Rect_nx1x2 = np.float32([[0, 0], [0 ,h-1], [w-1, h-1], [w-1,0]]).reshape(-1,1,2)
        HomoRect_nx1x2 = cv2.perspectiveTransform(Rect_nx1x2, HomoMatrix)
        cv2.polylines(QueryImg, [np.int32(HomoRect_nx1x2)], True, WHITE, thickness=3)
    else:
        print "Not enough matches are found - %d/%d" % (len(GoodMatches_SIFT), MIN_MATCH_COUNT)
        MatchesMask = None

    print 'invalid points: ', MatchesMask.count(1)
    MatchShowImg_SIFT = \
        BruteForceMatching.drawMatches(img1=ModelImg, keyPoints1=ModelKeyPoints_SIFT,
                                       img2=QueryImg, keyPoints2=QueryKeyPoints_SIFT,
                                       matches=GoodMatches_SIFT, matchesMask=MatchesMask)

    cv2.imshow('MatchShowImg_SIFT', MatchShowImg_SIFT)
    cv2.waitKey()
    cv2.destroyAllWindows()