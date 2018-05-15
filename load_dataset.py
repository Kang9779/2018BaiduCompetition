# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:16:48 2018
    数据集加载
@author: KANG
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from matplotlib.pyplot import imshow

def load_dataset(width = 197,height = 197):
    
    train_data = pd.read_csv('train.csv',header=None,encoding='gbk').values

    X_test = pd.read_csv('test.csv',header=None,encoding='gbk').values
    
    train_y = to_categorical(train_data[:,0]-1)
    
    train_x = train_data[:,1:]
    train_x = np.reshape(train_x,(-1,height,width,3))
    X_test = np.reshape(X_test,(-1,height,width,3))
    
    X_train,X_dev,y_train,y_dev = train_test_split(train_x,train_y,test_size=0.2)
    
    return X_train,y_train,X_dev,y_dev,X_test

#X_train,y_train,X_dev,y_dev,X_test = load_dataset()
#imshow(X_train[0])
#print(np.argmax(y_train[0])+1)
