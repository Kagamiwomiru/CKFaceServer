
# coding: utf-8

# In[12]:


import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.backend import tensorflow_backend as backend
from keras.preprocessing.image import ImageDataGenerator
import json

root_dir = "./face/"
batch_size=32
epochs=10
img_size = 224
classFile="./face/categories.json"


# In[2]:


def data_augmentation():
    #学習画像データを水増し（データ拡張）を行う
    mizumashi_data=ImageDataGenerator()
    mizumashi_generator=mizumashi_data.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)
    #テスト画像データを水増しする。
    val_datagen=ImageDataGenerator()
    val_gen=val_generator=val_datagen.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=False)
    valX,valy=val_gen.next()
    return (mizumashi_generator,val_generator,valX,valy)


# In[3]:


def build_model():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
	border_mode='same',
	input_shape=(img_size,img_size,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12))
    model.add(Activation('softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[4]:


def learning(model,mizumashi_generator,val_generator):
    tf_lg=keras.callbacks.TensorBoard(log_dir="./logs",write_grads=True)
    es_cb=keras.callbacks.EarlyStopping(monitor='val_loss',patience=0,verbose=1,mode='auto')

    log=[tf_lg,es_cb]
    history = model.fit_generator(mizumashi_generator,
                                  validation_data=val_generator,
                                  steps_per_epoch=mizumashi_generator.samples// batch_size,
                                  validation_steps=val_generator.samples // batch_size,
                                  epochs=epochs,callbacks=log,
                                  verbose=1)
    json_string=model.to_json()
    open('./face/face.json','w').write(json_string)
    hdf5_file = "./face/face-model.h5"
    model.save_weights(hdf5_file)
    return model



# In[8]:


def model_eval(model, X, y):
    score = model.evaluate(X, y,batch_size=batch_size)
    print('loss=', score[0])
    print('accuracy=', score[1])



# In[5]:


mizumashi_generator,val_generator,valX,valy=data_augmentation()


# In[6]:


model=build_model()


# In[7]:


learning(model,mizumashi_generator,val_generator)


# In[13]:


# class と indexの対応を逆にする
indices_to_class = dict((v, k)   for k, v in mizumashi_generator.class_indices.items())
print(indices_to_class)
f=open(classFile,'w')
json.dump(indices_to_class,f)


# In[14]:


model_eval(model, valX, valy)
backend.clear_session()

