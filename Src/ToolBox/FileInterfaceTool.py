#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__date__ = '22/01/2016'
__version__ = 2.0

import yaml
import pickle as pk
import cv2
import os
import numpy as np

def loadYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return  yaml.load(stream=file)

def loadAllYaml(fileName, method='r'):
    with open(fileName, method) as file:
        return yaml.load_all(stream=file)

def dumpYaml(data, fileName, method='w'):
    with open(fileName, method) as file:
        yaml.dump(data=data, stream=file)

def dumpAllYaml(data, fileName, method='w'):
    with open(fileName, method) as file:
        yaml.dump_all(documents=data, stream=file)

def loadPk(fileName, method='r'):
    with open(fileName, method) as File:
        return pk.load(File)

def dumpPk(data, fileName, method='w'):
    with open(fileName, method) as File:
        pk.dump(obj=data, file=File)

def isExist(fileName):
    return os.path.exists(fileName)

def createFolder(path, mode=0777, recursion=True):
    if not isExist(path):
        if recursion:
            os.makedirs(name=path, mode=mode)
        else:
            os.mkdir(path=path, mode=mode)



if __name__ == '__main__':
    createFolder(path='a/b/')