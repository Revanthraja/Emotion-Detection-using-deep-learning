# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 09:10:31 2022

@author: Toshiba

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tensorflow import keras
from keras.preprocessing.image import load_img ,img_to_array
from  keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Conv2D,MaxPooling2D,Activation,Dropout,Flatten,GlobalAveragePooling2D,BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
#from keras.optimizers import Adam,SGD,RMSprop

picture_size=48
folder_path='C:/Users/Toshiba/Downloads/images'
"""
expression='disgust'
plt.figure(figsize=(12,12))
for i in range(1,10,1):
    plt.subplot(3,3,i)
    img=load_img(folder_path+'train/'+expression+'/'+os.listdir(folder_path+'train/'+expression)[i],target_size=(picture_size,picture_size))
    plt.imshow(img)
plt.show()

"""
batch_size=128

datagen_train=ImageDataGenerator()
datagen_val=ImageDataGenerator()
train_set=datagen_train.flow_from_directory('C:/Users/Toshiba/Downloads/images/train',
                                            target_size=(picture_size,picture_size),
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            shuffle=True)
test_set=datagen_train.flow_from_directory('C:/Users/Toshiba/Downloads/images/validation',
                                            target_size=(picture_size,picture_size),
                                            color_mode='grayscale',
                                            class_mode='categorical',
                                            shuffle=False)


no_of_classes=7

model=Sequential()

model.add(Conv2D(64,(3,3),padding='same',input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(128,(5,5),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(no_of_classes,activation='softmax'))

model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=['accuracy'])
#model.summary()

checkpoint=ModelCheckpoint('./model1.h5',mointer='val_acc',varbose=1,save_best_only=True,mode='max')
earlystopping=EarlyStopping(monitor='val_loss',
    min_delta=0,
    patience=3,
    verbose=1,
    restore_best_weights=True)

redue_learningrate=ReduceLROnPlateau(
    monitor='val_lass',
    factor=0.2,
    patience=3,
    verbose=1,
    min_delta=0.0001)

callbacks_list=[earlystopping,redue_learningrate]

epochs=3

history=model.fit_generator(generator=train_set,
                  steps_per_epoch=train_set.n//train_set.batch_size,
                  epochs=epochs,
                  validation_data=test_set,
                  validation_steps=test_set.n//test_set.batch_size,
                  callbacks=callbacks_list)
model.save('model.h5')

