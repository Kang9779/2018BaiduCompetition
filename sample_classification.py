# -*- coding: utf-8 -*-
"""
Created on Sat May  5 09:49:54 2018
    样本归类输出到对应文件夹
@author: KANG
"""
import numpy as np
import pandas as pd
import cv2 as cv
import os

def sample_classification():
    filedir = './datasets/'
    jpg_trainSet = pd.read_csv(filedir+'train.txt',header=None,delim_whitespace=True,encoding='gbk')
    jpg_testSet = pd.read_csv(filedir+'test.txt',header=None,delim_whitespace=True,encoding='gbk')
    
    jpg_trainName = jpg_trainSet.loc[:,0].tolist()
    jpg_trainLabel = jpg_trainSet.loc[:,1].tolist()
    
    for index,jpgname in enumerate(jpg_trainName):
        img = cv.imread(filedir+'train/'+ jpgname)
        path = filedir + 'train_new/'
        if(not os.path.exists(path+str(jpg_trainLabel[index]))):
            os.makedirs(path+str(jpg_trainLabel[index]))
            cv.imwrite(path+str(jpg_trainLabel[index])+'/'+jpgname,img)    
        else:
            cv.imwrite(path+str(jpg_trainLabel[index])+'/'+jpgname,img)
#sample_classification()





