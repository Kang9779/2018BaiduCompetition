# -*- coding: utf-8 -*-
"""
Created on Wed May  2 11:20:32 2018

@author: KANG
"""
import pandas as pd
import numpy as np
import data_preprocessing as dp

import cv2 as cv 

'''
    绝对路径一般使用：'D:\\user\\ccc.txt'
    相对路径一般使用:'./datasets/a.txt','b.txt'
'''

def file2csv(filedir='./datasets/',width=197,height=197):
    train_file = pd.read_csv(filedir+'train.txt',header=None,
                             delim_whitespace=True,encoding='gbk')
    train_file.columns = ['name','label']
    test_file = pd.read_csv(filedir+'test.txt',header=None,encoding='gbk')
    test_file.columns = ['name']

    train_x = []
    train_y = train_file['label']
    test_x = []
    for jpgname in list(train_file['name']):
        img = cv.imread(filedir+'train/'+jpgname)
        img = dp.img_preprocessing(img,width,height)
        img = np.reshape(img,(1,width*height*3))
        train_x.append(img)
        
    for jpgname in list(test_file['name']):
        img = cv.imread(filedir+'test/'+jpgname)
        img = dp.img_preprocessing(img,width,height)
        img = img.reshape(1,width*height*3)
        test_x.append(img)
        
    train_x = np.concatenate(train_x,axis=0)
    test_x = np.concatenate(test_x,axis=0)
    
    train_pd = pd.DataFrame(train_x)
    train_label = pd.DataFrame(train_y)
    train_pd = pd.concat([train_label,train_pd],axis=1)
    
    test_xpd = pd.DataFrame(test_x)
    
    #文件输出
    train_pd.to_csv('train.csv',index=False,header=False,encoding='gbk')
    test_xpd.to_csv('test.csv',index=False,header = False,encoding='gbk')
    print("=====================DONE=================================")
    return train_x,test_x,train_y

#train_x,test_x,train_y = file2csv()
