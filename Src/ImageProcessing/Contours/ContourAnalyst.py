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


class ContourAnalyst(object):
    def __init__(self):
        object.__init__(self)

    @classmethod
    def getConvexityDefects(cls, contour):
        Hull = cv2.convexHull(contour, returnPoints = False)
        Defects = cv2.convexityDefects(contour, Hull)
        return Defects

    @classmethod
    def isPointInContour(cls, contour, point):
        """
        it finds whether the point is inside or outside or on the contour
        (it returns +1, -1, 0 respectively).
        """
        Point = np.array(point).reshape(-1).tolist()
        return cv2.pointPolygonTest(contour, Point, False)

    @classmethod
    def getDistance(cls, contour, point):
        Point = np.array(point).reshape(-1).tolist()
        return cv2.pointPolygonTest(contour, Point, True)

    @classmethod
    def matchContour(cls, contour1, contour2, method=1):
        """
        method:   1 - use I1 formula
                  2 - use I2 formula
                  3 - use I3 formula
        """
        # opencv
        # return cv2.matchShapes(contour1, contour2, method, 0.0)

        # DIY
        HuMoment1 = cv2.HuMoments(m=cv2.moments(contour1.astype(np.float32)))
        HuMoment2 = cv2.HuMoments(m=cv2.moments(contour2.astype(np.float32)))
        HuMoment1[abs(HuMoment1)<1e-9] = 0
        HuMoment2[abs(HuMoment2)<1e-9] = 0
        m1 = np.sign(HuMoment1) * np.log(np.abs(HuMoment1))
        m2 = np.sign(HuMoment2) * np.log(np.abs(HuMoment2))
        Valid1 = np.logical_not(np.isnan(m1))
        Valid2 = np.logical_not(np.isnan(m2))
        Valid = np.logical_and(Valid1, Valid2)
        # Valid = np.array([True]*7).reshape(HuMoment1.shape)
        if method == 1:
            DisSimilarity = np.sum(np.abs(1/m1[Valid] - 1/m2[Valid]))
        elif method == 2:
            DisSimilarity = np.sum(m1[Valid] - m2[Valid])
        elif method == 3:
            DisSimilarity = np.max(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
        elif method == 4:
            DisSimilarity = np.mean(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
        else:
            raise ValueError, 'method code error.'
        return DisSimilarity

    @classmethod
    def getAspectRatio(cls, contour):
        [_, _, w, h] = cls.getRoi_xywh(contour)
        return float(w) / h

    @classmethod
    def getExtentRatio(cls, contour):
        Area = cls.getArea(contour)
        [_, _, w, h] = cls.getRoi_xywh(contour)
        RectArea = w * h
        return float(Area) / RectArea

    @classmethod
    def getSolidityRatio(cls, contour):
        Area = cls.getArea(contour)
        Hull = cls.getConvexHull(contour)
        HullArea = cls.getArea(Hull)
        return float(Area) / HullArea

    @classmethod
    def getEquivalentDiameter(cls, contour):
        Area = cls.getArea(contour)
        return np.sqrt(4*Area / np.pi)

    @classmethod
    def getOrientation(cls, contour):
        _, _, _, Angle_rad = cls.fitEllipse(contour)
        return Angle_rad

    @classmethod
    def getExtremePoints(cls, contour):
        LeftMost_2x1   = np.array(contour[contour[:,:,0].argmin()][0]).reshape(2, 1)
        RightMost_2x1  = np.array(contour[contour[:,:,0].argmax()][0]).reshape(2, 1)
        TopMost_2x1    = np.array(contour[contour[:,:,1].argmin()][0]).reshape(2, 1)
        BottomMost_2x1 = np.array(contour[contour[:,:,1].argmax()][0]).reshape(2, 1)
        return LeftMost_2x1, RightMost_2x1, TopMost_2x1, BottomMost_2x1

    @classmethod
    def cvtPoints2Contour(cls, points_2xn):
        return points_2xn.T.reshape(-1, 1, 2).astype(np.int)

    @classmethod
    def isConvex(cls, contour):
        return cv2.isContourConvex(contour)

    @classmethod
    def getMoment(cls, contour):
        return cv2.moments(contour)

    @classmethod
    def getArea(cls, contour):
        return cv2.contourArea(contour)

    @classmethod
    def getCentroid(cls, contour):
        Moment = cls.getMoment(contour)
        try:
            CentroidPt_2x1 = np.array([[Moment['m10'] / Moment['m00']],
                                       [Moment['m01'] / Moment['m00']]])
        except ZeroDivisionError:
            CentroidPt_2x1 = contour.mean(0).reshape(2, 1)
        return CentroidPt_2x1

    @classmethod
    def getArcLenth(cls, contour, closed=True):
        return cv2.arcLength(curve=contour, closed=closed)

    @classmethod
    def approxPolyDP(cls, contour, approxPercent):
        MaxDis = approxPercent * cls.getArcLenth(contour)
        return cv2.approxPolyDP(curve=contour, epsilon=MaxDis, closed=True)

    @classmethod
    def getConvexHull(cls, contour):
        return cv2.convexHull(contour)

    @classmethod
    def getRoi_xywh(cls, contour):
        return cv2.boundingRect(contour)

    @classmethod
    def getRotatedRoi_xywh(cls, contour):
        Rect = cv2.minAreaRect(contour)
        Box = cv2.boxPoints(Rect)
        return np.int(Box)

    @classmethod
    def fitEnclosingCircle(cls, contour):
        (x,y),radius = cv2.minEnclosingCircle(contour)
        Center_2x1 = np.array([[x],
                               [y]])
        return Center_2x1, radius

    @classmethod
    def fitLine(cls, contour):
        return cv2.fitLine(contour, cv.CV_DIST_L2, 0, 0.01, 0.01)

    @classmethod
    def fitEllipse(cls, contour):
        (x, y), (MajorAxisLength, MinorAxisLength), Angle_rad =cv2.fitEllipse(contour)
        Center_2x1 = np.array([[x],
                               [y]])
        return Center_2x1, MajorAxisLength, MinorAxisLength, Angle_rad


if __name__ == '__main__':
    # SrcImg = cv2.imread('../../../Datas/Numbers.png')
    # SrcImg = cv2.imread('../../../Datas/CirclesInCircle.png')
    SrcImg = cv2.imread('../../../Datas/Line.png')
    GrayImg = cv2.cvtColor(src=SrcImg, code=cv2.COLOR_BGR2GRAY)
    _, BinImg = cv2.threshold(src=GrayImg, thresh=0, maxval=255, type=cv2.THRESH_OTSU)

    Contours, Hierarchy = \
        cv2.findContours(image=BinImg.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, offset=None)

    print 'Moment: ', ContourAnalyst.getMoment(Contours[0])
    print ContourAnalyst.getArea(Contours[0])
    print 'Centroid: ', ContourAnalyst.getCentroid(Contours[0])

    cv2.imshow('Gray', GrayImg)
    cv2.imshow('Bin', BinImg)
    cv2.waitKey()

