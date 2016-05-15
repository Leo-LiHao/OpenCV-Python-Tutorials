#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = 1.0
__date__ = 22/12/2015

import sys
sys.path.append("../../")

import cv2
try:
    import cv2.cv as cv
    OPENCV_VERSION = '2.0+'
except:
    print 'can not import cv2.cv'
    print 'cv2.cv is removed in opencv3.0+'
    OPENCV_VERSION = '3.0+'
import numpy as np
import threading
import time
import logging


class OpencvImportError(Exception):
    pass

class CameraSettingError(Exception):
    pass

class KeyBoardQuitSignal(Exception):
    pass

class CameraCapture(object):
    Logger = logging.getLogger('Camera')
    if '2.0+' == OPENCV_VERSION:
        FRAME_WIDTH     = cv.CV_CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT    = cv.CV_CAP_PROP_FRAME_HEIGHT
        POS_MSEC        = cv.CV_CAP_PROP_POS_MSEC
        POS_FRAMES      = cv.CV_CAP_PROP_POS_FRAMES
        AVI_RATIO       = cv.CV_CAP_PROP_POS_AVI_RATIO
        FPS             = cv.CV_CAP_PROP_FPS
        FOURCC          = cv.CV_CAP_PROP_FOURCC
        FRAME_COUNT     = cv.CV_CAP_PROP_FRAME_COUNT
        FORMAT          = cv.CV_CAP_PROP_FORMAT
        MODE            = cv.CV_CAP_PROP_MODE
        BRIGHTNESS      = cv.CV_CAP_PROP_BRIGHTNESS
        CONTRAST        = cv.CV_CAP_PROP_CONTRAST
        SATURATION      = cv.CV_CAP_PROP_SATURATION
        HUE             = cv.CV_CAP_PROP_HUE
        GAIN            = cv.CV_CAP_PROP_GAIN
        EXPOSURE        = cv.CV_CAP_PROP_EXPOSURE
        CONVERT_RGB     = cv.CV_CAP_PROP_CONVERT_RGB
        RECTIFICATION   = cv.CV_CAP_PROP_RECTIFICATION
    elif '3.0+' == OPENCV_VERSION:
        FRAME_WIDTH     = cv2.CAP_PROP_FRAME_WIDTH
        FRAME_HEIGHT    = cv2.CAP_PROP_FRAME_HEIGHT
        POS_MSEC        = cv2.CAP_PROP_POS_MSEC
        POS_FRAMES      = cv2.CAP_PROP_POS_FRAMES
        AVI_RATIO       = cv2.CAP_PROP_POS_AVI_RATIO
        FPS             = cv2.CAP_PROP_FPS
        FOURCC          = cv2.CAP_PROP_FOURCC
        FRAME_COUNT     = cv2.CAP_PROP_FRAME_COUNT
        FORMAT          = cv2.CAP_PROP_FORMAT
        MODE            = cv2.CAP_PROP_MODE
        BRIGHTNESS      = cv2.CAP_PROP_BRIGHTNESS
        CONTRAST        = cv2.CAP_PROP_CONTRAST
        SATURATION      = cv2.CAP_PROP_SATURATION
        HUE             = cv2.CAP_PROP_HUE
        GAIN            = cv2.CAP_PROP_GAIN
        EXPOSURE        = cv2.CAP_PROP_EXPOSURE
        CONVERT_RGB     = cv2.CAP_PROP_CONVERT_RGB
        RECTIFICATION   = cv2.CAP_PROP_RECTIFICATION
    else:
        raise OpencvImportError, 'import cv2.cv error!!!'

    def __init__(self, id):
        """
        camera init
        :param id: camera ID
        :return:
        """
        # assert isinstance(id, int), 'id must be int'
        # assert id>=0,               'id must >= 0'
        object.__init__(self)

        self.__CamId = id
        self.__Ready = False
        self.__Param = {}
        self.__Capture = None

    @property
    def Ready(self):
        return self.__Ready

    def __open(self, resolution, timeout=5):
        self.Logger.debug('opening %r' % self.__CamId)
        Start = time.time()
        self.__Capture = cv2.VideoCapture(self.__CamId)
        if not self.__Capture.isOpened():
            self.Logger.error('can not open [ID]%d camera!' % self.__CamId)
            raise CameraSettingError, 'open camera error!'

        if resolution is not None:
            self.Logger.debug('resolution: %s' % str(resolution))
            while True:
                time.sleep(0.01)
                Width = self.__Capture.get(self.FRAME_WIDTH)
                Height = self.__Capture.get(self.FRAME_HEIGHT)
                if Width == resolution[0] and Height == resolution[1]:
                    self.Logger.debug('resolution set ok')
                    break
                else:
                    time.sleep(0.01)
                    self.__Capture.set(self.FRAME_WIDTH, resolution[0])
                    self.__Capture.set(self.FRAME_HEIGHT, resolution[1])
                # check timeout
                if time.time() - Start > timeout:
                    self.Logger.error('time out')
                    raise CameraSettingError, 'camera resolution setting error'

        # self.__Ready = True
        while True:
            time.sleep(0.1)
            Ret, _ = self.__Capture.read()
            if Ret:
                # self.__Ready = True
                break
            # check timeout
            if time.time() - Start > timeout:
                self.Logger.error('time out')
                raise CameraSettingError, 'camera resolution setting error'
        self.__Ready = True
        self.Logger.debug('camera opened')

    def open(self, resolution=None, thread=False):
        if thread:
            MyThread = threading.Thread(target=self.__open, args=(resolution, ))
            MyThread.start()
        else:
            self.__open(resolution=resolution)

    def __config(self, resolution=(1280, 720), hue=0.5, saturation=0.5, brightness=0.5, contrast=0.5):
        self.__Ready = False
        self.__Capture.set(self.FRAME_WIDTH, resolution[0])
        self.__Capture.set(self.FRAME_HEIGHT, resolution[1])
        self.__Capture.set(self.HUE, hue)
        self.__Capture.set(self.SATURATION, saturation)
        self.__Capture.set(self.BRIGHTNESS, brightness)
        self.__Capture.set(self.CONTRAST, contrast)
        self.__Ready = True

    def config(self, resolution=(1280, 720), hue=0.5, saturation=0.5, brightness=0.5, contrast=0.5, thread=False):
        assert self.__Capture is not None, 'self.__Capture can not be none!!!'
        if thread:
            MyThread = threading.Thread(target=self.__config, args=(resolution, hue, saturation, brightness, contrast))
            MyThread.start()
        else:
            self.__config(resolution=resolution, hue=hue, saturation=saturation, brightness=brightness, contrast=contrast)

    def get(self, propId):
        return self.__Capture.get(propId)

    def set(self, propId, value):
        return self.__Capture.set(propId, value)

    def getParams(self):
        assert self.__Capture is not None, 'self.__Capture can not be None!!!'
        if not self.__Capture.isOpened() or not self.__Ready:
            return None
        self.__Param['HEIGHT']          = self.__Capture.get(self.FRAME_HEIGHT)
        self.__Param['WIDTH']           = self.__Capture.get(self.FRAME_WIDTH)
        self.__Param['POS_MSEC']        = self.__Capture.get(self.POS_MSEC)
        self.__Param['POS_FRAMES']      = self.__Capture.get(self.POS_FRAMES)
        self.__Param['POS_AVI_RATIO']   = self.__Capture.get(self.AVI_RATIO)
        self.__Param['FPS']             = self.__Capture.get(self.FPS)
        self.__Param['FOURCC']          = self.__Capture.get(self.FOURCC)
        self.__Param['FRAME_COUNT']     = self.__Capture.get(self.FRAME_COUNT)
        self.__Param['FORMAT']          = self.__Capture.get(self.FORMAT)
        self.__Param['MODE']            = self.__Capture.get(self.MODE)
        self.__Param['BRIGHTNESS']      = self.__Capture.get(self.BRIGHTNESS)
        self.__Param['CONTRAST']        = self.__Capture.get(self.CONTRAST)
        self.__Param['SATURATION']      = self.__Capture.get(self.SATURATION)
        self.__Param['HUE']             = self.__Capture.get(self.HUE)
        self.__Param['GAIN']            = self.__Capture.get(self.GAIN)
        self.__Param['EXPOSURE']        = self.__Capture.get(self.EXPOSURE)
        self.__Param['CONVERT_RGB']     = self.__Capture.get(self.CONVERT_RGB)
        # self.__Param['WHITE_BALANCE'] = self.__Capture.get(self.WHITE_BALANCE)
        self.__Param['RECTIFICATION']   = self.__Capture.get(self.RECTIFICATION)
        return self.__Param

    def isopened(self):
        return self.__Capture.isOpened()

    def takePhoto(self):
        if not self.__Ready:
            return None
        self.isCatch, self.img = self.__Capture.read()
        if self.isCatch:
            return self.img
        return None

    def release(self, thread=False):
        if thread:
            MyThread = threading.Thread(target=self.__release, args=())
            MyThread.start()
        else:
            self.__release()

    def __release(self):
        self.Logger.debug('release.. %d' % self.__CamId)
        self.__Ready = False
        if self.__Capture is not None:
            if self.__Capture.isOpened():
                self.__Capture.release()
        self.Logger.debug('released %d' % self.__CamId)

    def testFrameRate(self):
        if not self.__Ready:
            return None
        else:
            UseTime = []
            Start = time.time()
            while True:
                Now = time.time()
                PassTime = Now - Start
                if PassTime > 3:
                    break
                img = self.takePhoto()
                if img is not None:
                    UseTime.append(time.time()-Start)

            TimeEsp = np.array(UseTime[1:]) - np.array(UseTime[:-1])
            FrameRate = 1.0 / TimeEsp[1:].mean()
            return FrameRate

    @classmethod
    def __openCams(cls, cameraCaptureList, resolutionList):
        for i in range(len(cameraCaptureList)):
            # cls.Logger.debug('opening %d' % i)
            cameraCaptureList[i].open(resolution=resolutionList[i], thread=False)
            time.sleep(0.1)
            cls.Logger.debug('opened')

    @classmethod
    def openCams(cls, cameraCaptureList, resolutionList, thread=True):
        assert isinstance(cameraCaptureList, list), 'cameraCaptureList must be list'
        assert isinstance(resolutionList, list),    'resolutionList must be list'
        assert len(resolutionList) == len(cameraCaptureList), 'the length of 2 lists must equaled'
        assert cameraCaptureList, 'cameraCaptureList should have at least one element'
        assert isinstance(cameraCaptureList[0], CameraCapture), 'the element os cameraCaptureList must be [class] CameraCapture'

        if thread:
            MyThread = threading.Thread(target=cls.__openCams, args=(cameraCaptureList, resolutionList))
            MyThread.start()
        else:
            cls.__openCams(cameraCaptureList=cameraCaptureList, resolutionList=resolutionList)

    @classmethod
    def __releaseCams(cls, cameraCaptureList, NOP):
        for i in range(len(cameraCaptureList)):
            # cls.Logger.debug('opening %d' % i)
            cameraCaptureList[i].release(thread=False)
            time.sleep(0.1)
            # cls.Logger.debug('released')

    @classmethod
    def releaseCams(cls, cameraCaptureList, NOP=None, thread=True):
        assert isinstance(cameraCaptureList, list), 'cameraCaptureList must be list'
        assert cameraCaptureList, 'cameraCaptureList should have at least one element'
        assert isinstance(cameraCaptureList[0], CameraCapture), 'the element os cameraCaptureList must be [class] CameraCapture'

        if thread:
            MyThread = threading.Thread(target=cls.__releaseCams, args=(cameraCaptureList, NOP))
            MyThread.start()
        else:
            cls.__releaseCams(cameraCaptureList=cameraCaptureList, NOP=NOP)


# if __name__ == '__main__':
#     Cam = CameraCapture(4)
#     # Cam.open(thread=False, resolution=(640, 480))
#     Cam.open(thread=False, resolution=(1600, 1200))
#     # CameraCapture.openCams([Cam], [None], True)
#     # Cam.open(thread=False, resolution=(1600, 1200))
#     num = 0
#     T0 = time.time()
#     count = 0
#     while True:
#         # print 'num: ', num
#         # num += 1
#         time.sleep(0.1)
#         Img = Cam.takePhoto()
#         if Img is None:
#             PassTime = time.time() - T0
#             if PassTime > count:
#                 print 'Passed ', PassTime, 'second'
#                 count += 1
#             if count > 10:
#                 break
#             continue
#
#         T0 = time.time()
#         cv2.imshow('img', Img)
#         Key = chr(cv2.waitKey(10) & 255)
#         if 'q' == Key:
#             break
#
#     print Cam.set(propId=CameraCapture.BRIGHTNESS, value=0)
#     print Cam.set(propId=CameraCapture.BRIGHTNESS, value=0)
#     print 'frame_height', Cam.get(propId=CameraCapture.FRAME_HEIGHT)
#     print 'FrameRate: ', Cam.testFrameRate()
#     print 'Param:\n', Cam.getParams()
#     Cam.release()
#     # print 'aaaaaa'
            

# # # ---------------------------------- Stereo Cam ---------------------------------- #
# if __name__ == '__main__':
#     Cam5 = CameraCapture(2)
#     Cam7 = CameraCapture(4)
#     Wait5FrameFlag = 5
#     CameraCapture.openCams(cameraCaptureList=[Cam5, Cam7],
#                            resolutionList=[(1600, 1200), (1600, 1200)], thread=False)
#
#     while True:
#         SrcImg0 = Cam5.takePhoto()
#         SrcImg1 = Cam7.takePhoto()
#
#         if None in (SrcImg0, SrcImg1):
#             NowTime = time.time()
#             if PreTime is None:
#                 PreTime = NowTime - 0.51
#             if NowTime - PreTime >= 0.5:
#                 PreTime = NowTime
#             continue
#
#         if Wait5FrameFlag > 0:
#             if Wait5FrameFlag == 5:
#                 print 'start: ', time.time()
#                 startWait = time.time()
#             SaveStart = time.time()
#             cv2.imwrite('../../Datas/Img/CatchTail/waitFrameImgTest_Cam0_'+str(Wait5FrameFlag)+'.png', SrcImg0)
#             cv2.imwrite('../../Datas/Img/CatchTail/waitFrameImgTest_Cam1_'+str(Wait5FrameFlag)+'.png', SrcImg1)
#             print 'Save time: ', time.time() - SaveStart
#             Wait5FrameFlag -= 1
#             if Wait5FrameFlag == 0:
#                 print 'end: ', time.time()
#                 print '5 Frame: ', time.time() - startWait
#             continue
#
#
#         cv2.imshow('img0', SrcImg0)
#         cv2.imshow('img1', SrcImg1)
#         Key = chr(cv2.waitKey(5) & 0xff)
#         if 'q' == Key:
#             break
#     print 'cam 5 FrameRate: ', Cam5.testFrameRate()
#     print 'cam 7 FrameRate: ', Cam7.testFrameRate()

# # ---------------------------------- 2 Pair of Stereo Cam ---------------------------------- #
# if __name__ == '__main__':
#     Cam5 = CameraCapture(2)
#     Cam7 = CameraCapture(4)
#     Cam3 = CameraCapture(0)
#     Cam4 = CameraCapture(1)
#     CamPairList = [[Cam3, Cam4], [Cam5, Cam7]]
#
#     PairNum = 0
#     TimeNum = 0
#     while True:
#         start = time.time()
#         print 'the ', TimeNum, ' time'
#         TimeNum += 1
#         if PairNum > 1:
#             PairNum = 0
#         CamPair = CamPairList[PairNum]
#         print 'now open...'
#         print 'camera pair: ', PairNum
#         # use new way
#         CameraCapture.setCamsResolution(cameraCaptureList=CamPair, resolutionList=[(1600, 1200), (1600, 1200)], thread=True)
#         # old method
#         # for Cam in CamPair:
#         #     # Cam.open(resolution=(1600, 1200), thread=False)
#         #     Cam.open(resolution=(1600, 1200), thread=True)
#         # end
#         while True:
#             # PassTime = time.time() - start
#             # print 'PassTime: ', PassTime
#             if time.time() - start > 10:
#                 for Cam in CamPair:
#                     Cam.release()
#                     time.sleep(1)
#                 break
#             Img0 = CamPair[0].takePhoto()
#             Img1 = CamPair[1].takePhoto()
#             if None in (Img0, Img1):
#                 continue
#             cv2.imshow('Img0_'+str(PairNum), cv2.resize(Img0, (Img0.shape[1]/4, Img0.shape[0]/4)))
#             cv2.imshow('Img1_'+str(PairNum), cv2.resize(Img1, (Img1.shape[1]/4, Img1.shape[0]/4)))
#             Key = chr(cv2.waitKey(5) & 0xff)
#             if 'q' == Key:
#                 raise KeyBoardQuitSignal, 'pressed [q] in program'
#         PairNum += 1

if __name__ == '__main__':
    Cam = CameraCapture(2)
    # Cam2 = CameraCapture(1)
    Cam.open(resolution=[1600, 1200])
    # Cam.open(resolution=(1280, 720))
    # CameraCapture.openCams([Cam], [[1280, 720]])
    while True:
        Img = Cam.takePhoto()
        if Img is not None:
            cv2.imshow('Img', Img)
            Key = chr(cv2.waitKey(0) & 255)
            if Key == 'q':
                break
    print Cam.testFrameRate()

