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


RED   = (0, 0, 255)
BLUE  = (255, 0, 0)
GREEN = (0, 255, 0)


def drawMatches(img1, keyPoints1, img2, keyPoints2, matches, matchesMask=None,
                matchColor=(0, 255, 0), missColor=(0, 0, 255), circleRadius=4, circleThickness=2,
                lineColor=(255, 0, 0), lineThickness=1):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]
    OutputImg = np.zeros((max(rows1, rows2), cols1+cols2, 3), dtype=np.uint8)
    if img1.ndim == 3:
        OutputImg[:rows1, :cols1, :] = img1
    elif img1.ndim == 2:
        OutputImg[:rows1, :cols1, :] = np.dstack((img1, img1, img1))
    else:
        raise ValueError, 'img1 data error: img1.ndim is not 2 or 3'
    if img2.ndim == 3:
        OutputImg[:rows2, cols1:, :] = img2
    elif img2.ndim == 2:
        OutputImg[:rows2, cols1:, :] = np.dstack((img2, img2, img2))
    else:
        raise ValueError, 'img1 data error: img1.ndim is not 2 or 3'

    for idx, match in enumerate(matches):
        if matchesMask is not None:
            if matchesMask[idx] == 0:
                continue
        # Get the matching key points for each of the images
        img1Idx = match.queryIdx
        img2Idx = match.trainIdx
        (x1, y1) = keyPoints1[img1Idx].pt
        (x2, y2) = keyPoints2[img2Idx].pt

        cv2.circle(OutputImg, (int(x1),int(y1)), circleRadius, matchColor, circleThickness)
        cv2.circle(OutputImg, (int(x2)+cols1,int(y2)), circleRadius, matchColor, circleThickness)
        cv2.line(OutputImg, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), lineColor, lineThickness)
    return OutputImg


if __name__ == '__main__':
    ModelImgPath = '../../../Datas/box.png'
    QueryImgPath = '../../../Datas/box_in_scene.png'
    ModelImg = cv2.imread(ModelImgPath)
    QueryImg = cv2.imread(QueryImgPath)
    ModelGrayImg = cv2.cvtColor(src=ModelImg, code=cv2.COLOR_BGR2GRAY)
    QueryGrayImg = cv2.cvtColor(src=QueryImg, code=cv2.COLOR_BGR2GRAY)

    # -------------- Detector ORB --------------- #
    FeaturePointsNum = 500
    PyramidScale = 1.2
    PyramidLevel = 8
    PatchSize = 31           # size of the patch used by the oriented BRIEF descriptor
    # PatchSize = 111           # size of the patch used by the oriented BRIEF descriptor
    EdgeThresh = PatchSize
    FirstLevel = 0           # default
    OrientedBRIEFPointNum = 2  # 2, 3, 4
    ScoreType = cv2.ORB_HARRIS_SCORE
    # ScoreType = cv2.ORB_FAST_SCORE
    ORB = cv2.ORB(nfeatures=FeaturePointsNum, scaleFactor=PyramidScale, nlevels=PyramidLevel, edgeThreshold=EdgeThresh,
                  firstLevel=FirstLevel, WTA_K=OrientedBRIEFPointNum, scoreType=ScoreType, patchSize=PatchSize)

    # -------------- detect features --------------- #
    ModelKeyPoints = ORB.detect(ModelGrayImg, mask=None)
    ModelKeyPoints, ModelDescriptions = ORB.compute(ModelGrayImg, ModelKeyPoints)
    ModelShowImg = cv2.drawKeypoints(image=ModelImg, keypoints=ModelKeyPoints, color=RED)

    QueryKeyPoints = ORB.detect(QueryGrayImg, mask=None)
    QueryKeyPoints, QueryDescriptions = ORB.compute(QueryGrayImg, QueryKeyPoints)
    QueryShowImg = cv2.drawKeypoints(image=QueryImg, keypoints=QueryKeyPoints, color=RED)

    cv2.imshow('ModelShowImg', ModelShowImg)
    cv2.imshow('QueryShowImg', QueryShowImg)

    # -------------- matching --------------- #
    # use crossCheck instead of ratio check
    CrossCheckFlag = True
    BFMatcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=CrossCheckFlag)
    MatchResult = BFMatcher.match(queryDescriptors=ModelDescriptions, trainDescriptors=QueryDescriptions)
    # Sort them in the order of their distance.
    MatchResultSorted = sorted(MatchResult, key = lambda x:x.distance)
    MatchShowImg = drawMatches(img1=ModelImg, keyPoints1=ModelKeyPoints,
                               img2=QueryImg, keyPoints2=QueryKeyPoints, matches=MatchResultSorted[:50])
    cv2.imshow('MatchShowImg', MatchShowImg)
    cv2.waitKey()
    cv2.destroyAllWindows()


    # -------------- Detector SIFT --------------- #
    SIFT = cv2.SIFT()
    ModelKeyPoints, ModelDescriptions = SIFT.detectAndCompute(ModelGrayImg, mask=None)
    QueryKeyPoints, QueryDescriptions = SIFT.detectAndCompute(QueryGrayImg, mask=None)

    # -------------- matching --------------- #
    BFMatcher.setBool('crossCheck', False)
    BFMatcher.setBool('normType', cv2.NORM_L2)
    KnnMatchResult = BFMatcher.knnMatch(queryDescriptors=ModelDescriptions, trainDescriptors=QueryDescriptions, k=2)
    print 'len(KnnMatchResult)', len(KnnMatchResult)
    print 'len(MatchResult)', len(MatchResult)
    print 'QueryDescriptions.shape', QueryDescriptions.shape
    print 'ModelDescriptions.shape', ModelDescriptions.shape

    # Apply ratio test
    GoodMatches = []
    for m, n in KnnMatchResult:
        if m.distance < 0.75*n.distance:
            GoodMatches.append(m)
    print 'GoodMatches: ', len(GoodMatches)
    MatchShowImg = drawMatches(img1=ModelImg, keyPoints1=ModelKeyPoints,
                               img2=QueryImg, keyPoints2=QueryKeyPoints, matches=GoodMatches)
    cv2.imshow('MatchShowImg', MatchShowImg)
    cv2.waitKey()
    cv2.destroyAllWindows()