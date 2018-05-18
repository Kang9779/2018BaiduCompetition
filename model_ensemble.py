# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:54:16 2018
    模型融合与预测
@author: KANG
"""
from keras.applications.xception import Xception
import  inception_v3_L2  as my_inception
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,BatchNormalization,Dropout,GlobalMaxPooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,ThresholdedReLU
from keras import regularizers
from keras import backend as K
import load_dataset as ld
import pandas as pd
from keras import optimizers
from keras.utils import multi_gpu_model
import numpy as np
import datetime

def model_Xception():
    '''
        加载训练好的Xception模型
    '''
    base_model = Xception(include_top=False,weights='imagenet',input_shape=(197,197,3))
    x = base_model.output
    x = GlobalMaxPooling2D(name="max_pool")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(1024,kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(512,kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization(axis=-1)(x)
    predictions = Dense(100, activation='softmax',name='predictions')(x)
    model_Xception = Model(inputs=base_model.input, outputs=predictions).load_weights('model_xception.h5')
    return model_Xception

def model_Inception():
    '''
        加载训练好的Inception模型
    '''
    base_model = my_inception.InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalMaxPooling2D(name="max_pool")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(1024,kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(512,kernel_regularizer=regularizers.l2(0.01))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)
    predictions = Dense(100, activation='softmax',name='predictions')(x)
    model_Inception = Model(inputs=base_model.input, outputs=predictions).load_weights('model_inception.h5')
    return model_Inception

def model_evaluate(model_Inception,model_Xception,X_train,y_train):
    '''
        input:
            model_Inception : Inception 模型
            model_Xception : Xception 模型
            X_train: 训练集样本
            y_train:训练集标签
        output:
            模型融合后的精度
    '''
    result_xception = model_Xception.predict(X_train)
    result_inception = model_Inception.predict(X_train)
    result_label = []
    length = result_xception.shape[0]
    count = 0
    for index in range(length):
        xception_max = max(result_xception[index])
        xception_index = result_xception[index].index(xception_max)
        
        inception_max = max(result_inception[index])
        inception_index = result_inception[index].index(inception_max)
        if xception_max>inception_max:
            result_label.append(xception_index)
        else:
            result_label.append(inception_index)
        if result_label[index]==y_train[index]:
            count+=1
    accuracy = count/length * 100
    return accuracy

def model_predict(model_Inception,model_Xception,X_test):
    '''
        input:
            训练好的模型
        output:
            predict_label:对X_test样本预测
    '''
    predict_xception = model_Xception.predict(X_train)
    predict_inception = model_Inception.predict(X_train)
    predict_label = []
    length = predict_xception.shape[0]
    for index in range(length):
        xception_max = max(predict_xception[index])
        xception_index = predict_xception[index].index(xception_max)
        
        inception_max = max(predict_inception[index])
        inception_index = predict_inception[index].index(inception_max)
        if xception_max>inception_max:
            predict_label.append(xception_index)
        else:
            predict_label.append(inception_index)
    predict_label = predict_label + 1
    return predict_label


if __name__=='_main_':
    
    nowtime = datetime.datetime.now().strftime('%m%d-%H-%M')
    print("======================dataSet loading...=========================")
    X_train,y_train,X_dev,y_dev,X_test = ld.load_dataset()
    X_train = X_train/127.5 - 1
    X_dev = X_dev/127.5 - 1
    X_test = X_test/127.5 - 1
    
    model_Xception = model_Xception()
    model_Inception = model_Inception()
    
    print("=========================Model Envaluate========================")
    Accuracy = model_evaluate(model_Inception,model_Xception,X_train,y_train)
    print("ModelEnsemble Accuracy:"+str(Accuracy)+' %')
    
    print("========================ModelEnsemble predicting===============")
    predict_label = model_predict(model_Inception,model_Xception,X_test)
    results = pd.Series(predict_label)
    X_result = pd.read_csv('./datasets/test.txt',header=None,encoding='gbk')
    submission = pd.concat([X_result,results],axis = 1,)
    submission.to_csv("modelEnsemble_predict-"+nowtime+".csv",sep=" ",index=False,header=False)





    
