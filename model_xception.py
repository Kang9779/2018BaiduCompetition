# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:14 2018

@author: KANG
"""
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,BatchNormalization,Dropout
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
import load_dataset as ld
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import plot_model
from keras import optimizers
from keras.utils import multi_gpu_model
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os

print("==================xception readying========================")
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'# Running GPU Devices
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

base_model = Xception(include_top=False,weights='imagenet',input_shape=(197,197,3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = BatchNormalization(axis=-1)(x)
#x = Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
#x = Dropout(0.2)(x)
#x = BatchNormalization(axis=-1)(x)
#x = Dense(512,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
#x = Dropout(0.2)(x)
x = BatchNormalization(axis=-1)(x)
predictions = Dense(100, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = True
adam = optimizers.Adam(lr=0.002)
model = multi_gpu_model(model,gpus=2)
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])


def scheduler(epoch):
    if epoch<50:
        lr = 0.002
        return lr
    if epoch < 100:
        lr = 0.001
        return lr
    if epoch < 150:
        lr = 0.0005
        return lr
    return 0.00025

print("=============数据集加载中...================")
# train the model on the new data for a few epochs
X_train,y_train,X_dev,y_dev,X_test = ld.load_dataset()
#归一化
X_train = X_train/255.0 - 0.5
X_dev = X_dev/255.0 - 0.5
X_test = X_test/255.0 - 0.5
#分类类别数
num_classes = 100
batch_size = 64#...debug
steps_per_epoch = X_train.shape[0]
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
#                                            patience=2, 
#                                            verbose=1, 
#                                            factor=0.5, 
#                                            min_lr=0)
#learning_rate_reduction = LearningRateScheduler(scheduler)
learning_rate_reduction = LearningRateScheduler(scheduler)
traindatagen = ImageDataGenerator(horizontal_flip=False,
                                 rotation_range = 10,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range = 0.015,
                                 zoom_range = 0.015,
                                 fill_mode='nearest',cval=0.)
devdatagen = ImageDataGenerator(horizontal_flip=False,
                                 rotation_range = 10,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range = 0.015,
                                 zoom_range = 0.015,
                                 fill_mode='nearest',cval=0.)
    
traindatagen.fit(X_train)
devdatagen.fit(X_dev)

history = model.fit_generator(traindatagen.flow(X_train,y_train,batch_size=batch_size),
                        steps_per_epoch=2*steps_per_epoch//batch_size,
                        epochs=200,
                        validation_data=devdatagen.flow(X_dev,y_dev,batch_size=batch_size),
                        callbacks=[learning_rate_reduction])

preds = model.evaluate(X_dev,y_dev)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

fig,ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label='Training loss')
ax[0].plot(history.history['val_loss'],color='r',label='validation loss',axes=ax[0])
legend = ax[0].legend(loc='best',shadow=True)
ax[1].plot(history.history['acc'],color='b',label='Training accuracy')
ax[1].plot(history.history['val_acc'],color='r',label='validation accuracy')
legend = ax[1].legend(loc='best',shadow=True)
plt.savefig("Xception-0515.png")

print("==================预测生成...=====================")
results = model.predict(X_test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1) + 1

results = pd.Series(results)
X_result = pd.read_csv('./datasets/test.txt',header=None,encoding='gbk')
submission = pd.concat([X_result,results],axis = 1,)
submission.to_csv("xception_predict.csv",sep=" ",index=False,header=False)

f = open('Xceptiontrain_log.txt','w')
f.write("============================Training Log==================="+'\n')
for index in range(len(history.history['val_loss'])):
    f.write('val_loss:'+str(history.history['val_loss'][index])+'    '+'lr:'+str(history.history['lr'][index]) 
    +'    '+'val_acc:'+str(history.history['val_acc'][index])+'    '+'loss:'+str(history.history['loss'][index])+'\n')
f.close()

print("=================模型文件保存中...====================")
model.save("model_xception.h5")





