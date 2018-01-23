
# coding: utf-8

# In[1]:


import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import AveragePooling2D
from keras.utils import np_utils
from keras.applications.xception import Xception
from keras.optimizers import SGD
from keras import callbacks
from keras.backend import tensorflow_backend as backend
import json
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import shutil
import sys
args=sys.argv


# In[2]:


# 画像サイズ．ResNetを使う時は224
img_size = 224
batch_size = 32
#以下ディレクトリに入っている画像を読み込む
root_dir = "./face/"
#学習データを何周するか
epochs=10
#ログファイル
log_filepath="./logs/"
#学習したモデル
ModelWeightData="./face/face-model.h5"
ModelArcData="./face/face.json"
classFile="./face/categories.json"

#Imagegeneratorサンプル画像
Sample="./test/"
#tmpファイルを保存するディレクトリ
temp_dir="./result/tmpimg"
#python3 face_keras.py zoom $j
result_images="./result/"+args[1]+"/out"+args[2]+".png"


# In[3]:


#ImageGenerator
os.mkdir(temp_dir) 
test_data=ImageDataGenerator(featurewise_std_normalization=True,samplewise_std_normalization=True)
mizumashi_data=ImageDataGenerator(featurewise_std_normalization=True,samplewise_std_normalization=True)
val_datagen=ImageDataGenerator(featurewise_std_normalization=True,samplewise_std_normalization=True)


# In[4]:


#data_augmentation()
mizumashi_generator=mizumashi_data.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)
#テスト画像データを水増しする。

val_gen=val_generator=val_datagen.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=False)
valX,valy=val_gen.next()


# In[5]:


test_generator=test_data.flow_from_directory(save_to_dir=temp_dir,directory=Sample,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)


# In[6]:


for i in range(9):
    batch=test_generator.next()


# In[7]:


#生成した画像を3x3で描画
images = glob.glob(os.path.join(temp_dir,"*.png"))


# In[8]:


gs=gridspec.GridSpec(3,3)


# In[9]:


gs.update(wspace=0.1,hspace=0.1)


# In[10]:


for i in range(9):
        img=load_img(images[i])
        plt.subplot(gs[i])
        plt.imshow(img,aspect='auto')
        plt.axis("off")
        plt.savefig(result_images)
shutil.rmtree(temp_dir)


# In[11]:


#重みvをimagenetとすると、学習済みパラメータを初期値としてXceptionを読み込む。
#model_load()
base_model = Xception(weights='imagenet', include_top=False,
                     input_tensor=Input(shape=(img_size,img_size, 3)))
#base_model.summary()
x=base_model.output
#入力を平滑化
x=Flatten()(x)
#過学習防止
x=Dropout(.8)(x)


# In[12]:


#model_build()
# 最後の全結合層の出力次元はクラスの数(= mizumashi_generator.num_class)
predictions = Dense(mizumashi_generator.num_classes,kernel_initializer='glorot_uniform', activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
opt = SGD()


# In[13]:


#learning()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
tb_cb=keras.callbacks.TensorBoard(log_dir=log_filepath,histogram_freq=0)
cbks=[tb_cb]

history = model.fit_generator(mizumashi_generator,
                              validation_data=val_generator,
                              steps_per_epoch=mizumashi_generator.samples// batch_size,
                              validation_steps=val_generator.samples // batch_size,
                              epochs=epochs,callbacks=cbks,
                              verbose=1)

#モデルの構造と重みを保存。
json_string=model.to_json()
open(ModelArcData,'w').write(json_string)
model.save_weights(ModelWeightData)


# In[14]:


score = model.evaluate(valX, valy,batch_size=batch_size)
print('loss=', score[0])
print('accuracy=', score[1])

