# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:00:14 2018
    fine-tuning:
        激活函数尝试采用LeakRelu或者其他；
@author: KANG
"""
from Xception_myself import Xception
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D,BatchNormalization,Dropout,GlobalMaxPooling2D
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,ThresholdedReLU
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
import os
import datetime


print("==================xception readying========================")
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'# Running GPU Devices
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

base_model = Xception(include_top=False,weights='imagenet',input_shape=(197,197,3))

#%%
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

# =======================model and optimizer setting=======================
model = Model(inputs=base_model.input, outputs=predictions)
model = multi_gpu_model(model,gpus=4)

#========================Loading the datasets==============================
print("=============数据集加载中...================")
# train the model on the new data for a few epochs
X_train,y_train,X_dev,y_dev,X_test = ld.load_dataset()
#归一化
X_train = X_train/127.5 - 1
X_dev = X_dev/127.5 - 1
X_test = X_test/127.5 - 1
#分类类别数
num_classes = 100
batch_size = 32
steps_per_epoch = X_train.shape[0]
traindatagen = ImageDataGenerator(horizontal_flip=False,
                                 rotation_range = 20,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range = 0.015,
                                 zoom_range = 0.015,
                                 channel_shift_range=0.01,
                                 fill_mode='constant',cval=0.)
devdatagen = ImageDataGenerator(horizontal_flip=False,
                                 rotation_range = 20,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range = 0.015,
                                 zoom_range = 0.015,
                                 channel_shift_range=0.01,
                                 fill_mode='constant',cval=0.)

traindatagen.fit(X_train)
devdatagen.fit(X_dev)

print("=====================全局训练...=================================")
for layer in model.layers:
    layer.trainable = True
    
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            factor=0.2,
                                            patience=3,
                                            verbose=1,
                                            epsilon=1e-6,
                                            min_lr=0)
adam = optimizers.Adam(lr=0.0015)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
#learning_rate_reduction = LearningRateScheduler(scheduler)
history = model.fit_generator(traindatagen.flow(X_train,y_train,batch_size=batch_size),
                        steps_per_epoch=2*steps_per_epoch//batch_size,
                        epochs=100,
                        validation_data=devdatagen.flow(X_dev,y_dev,batch_size=batch_size),
                        callbacks=[learning_rate_reduction])

preds = model.evaluate(X_dev,y_dev)
print("全局训练：")
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

nowtime = datetime.datetime.now().strftime('%m%d-%H-%M')

fig,ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'],color='b',label='Training loss')
ax[0].plot(history.history['val_loss'],color='r',label='validation loss',axes=ax[0])
legend = ax[0].legend(loc='best',shadow=True)
ax[1].plot(history.history['acc'],color='b',label='Training accuracy')
ax[1].plot(history.history['val_acc'],color='r',label='validation accuracy')
legend = ax[1].legend(loc='best',shadow=True)
plt.savefig("xception-"+nowtime+"acc-"+str(preds[1])+".png")

#=======================预测===============================
print("==================预测生成...=====================")
results = model.predict(X_test)

resultsProp = pd.DataFrame(data=results)
resultsProp.to_csv("results"+nowtime+"acc-"+str(preds[1])+".csv",index=False,header=False)

results = np.argmax(results,axis = 1) + 1

results = pd.Series(results)
X_result = pd.read_csv('./datasets/test.txt',header=None,encoding='gbk')
submission = pd.concat([X_result,results],axis = 1,)
submission.to_csv("xception_predict-"+nowtime+"acc-"+str(preds[1])+".csv",sep=" ",index=False,header=False)

f = open('Xceptiontrain_log-'+nowtime+'.txt','w')
f.write("============================Training Log==================="+'\n')
for index in range(len(history.history['val_loss'])):
    f.write('val_loss:'+str(history.history['val_loss'][index])+'    '+'val_acc:'+str(history.history['val_acc'][index])+'    '+'loss:'+str(history.history['loss'][index])+'\n')
f.close()

print("=================模型文件保存中...====================")
model.save_weights("model_xception-"+nowtime+"acc-"+str(preds[1])+".h5")
