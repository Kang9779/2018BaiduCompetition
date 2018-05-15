# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:16:04 2018
    预处理
@author: KANG
"""
import cv2 as cv

def img_preprocessing(img,width,height):
    img = cv.resize(img,(width,height),interpolation=cv.INTER_LINEAR)
    return img

