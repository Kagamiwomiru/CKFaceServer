
# coding: utf-8

# In[577]:


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
import matplotlib.pyplot as plt
from matplotlib import gridspec
import shutil
import sys
args=sys.argv


# In[578]:


# 画像サイズ．ResNetを使う時は224
img_size = 224
batch_size = 32
#以下ディレクトリに入っている画像を読み込む
root_dir = "./face/"
#学習データを何周するか
epochs=50
#ログファイル
log_filepath="./logs/"
#学習したモデル
ModelWeightData="./model/face-model.h5"
ModelArcData="./model/face.json"
classFile="./model/categories.json"

#Imagegeneratorサンプル画像
Sample="./test/"
#tmpファイルを保存するディレクトリ
temp_dir="./result/tmpimg"
#python3 face_keras.py $i $j
#result_images="./result/"+args[1]+"/out"+args[2]+".png"
result_images="./result/out.png"


# In[579]:


#輝度をあげます
import MakeData as DA
#重複防止


# In[597]:


os.system('./initface.sh ')


# In[598]:
#水増し部分
if(args[1]=='0'):
    DA.high_cont('DA')
    DA.data_eraser(int(110/2))
elif(args[1]=='1'):
    DA.edge_detection('hg')
    DA.data_eraser(int(110/2))
elif (args[1]=='2'):
    DA.high_cont('DA')
    DA.data_eraser(int(110/2))
    DA.edge_detection('hg')
    DA.data_eraser(int(110/2))


# In[599]:


#ImageGenerator
# mizumashi_data=ImageDataGenerator(height_shift_range=0.2,zoom_range=0.2,shear_range=40) 
# val_datagen=ImageDataGenerator(height_shift_range=0.2,zoom_range=0.2,shear_range=40) 
mizumashi_data=ImageDataGenerator() 
val_datagen=ImageDataGenerator() 


# In[600]:


#data_augmentation()
mizumashi_generator=mizumashi_data.flow_from_directory(class_mode="categorical",directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)
#テスト画像データを水増しする。

val_gen=val_generator=val_datagen.flow_from_directory(class_mode="categorical",directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=False)
valX,valy=val_gen.next()


# In[601]:


# class と indexの対応を逆にする
indices_to_class = dict((v, k)   for k, v in mizumashi_generator.class_indices.items())
print(indices_to_class)
f=open(classFile,'w')
json.dump(indices_to_class,f)


# In[602]:


#重みvをimagenetとすると、学習済みパラメータを初期値としてXceptionを読み込む。
#model_load()
base_model = Xception( weights='imagenet',include_top=False,
                     input_tensor=Input(shape=(img_size,img_size ,3)))
#base_model.summary()
x=base_model.output
#入力を平滑化
x=Flatten()(x)
#過学習防止
x=Dropout(.4)(x)


# In[603]:


#model_build()
# 最後の全結合層の出力次元はクラスの数(= mizumashi_generator.num_class)
predictions = Dense(mizumashi_generator.num_classes,kernel_initializer='glorot_uniform', activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
opt = SGD(lr=0.05)


# In[604]:


#learning()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
tb_cb=keras.callbacks.TensorBoard(log_dir=log_filepath,histogram_freq=0)
es_cb=keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto')
cbks=[tb_cb,es_cb]

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


# In[605]:


score = model.evaluate(valX, valy,batch_size=batch_size)
print('loss=', score[0])
print('accuracy=', score[1])


# In[606]:


backend.clear_session()


# # ラズパイが死ぬのでサーバで認証作業させます
# 

# In[607]:


f=open(classFile,'w')
json.dump(indices_to_class,f)


# In[608]:


# coding:utf-8
import keras
import sys, os
import scipy
import scipy.misc
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Model
import collections #kerasと関係ないです。
import json
from keras.backend import tensorflow_backend as backend
import time as t
import re #正解判別
start_t=t.time()
#画像サイズ
imsize = (96, 96)
#人数
people=9
#認識したい画像のパスを指定する
# ./blog_testpic/xxx.jpg といった指定を意味する
testpic = "./Auth/"
#使用するモデルを指定する
keras_model = "./model/face.json"
keras_param = "./model/face-model.h5"
#合格点（0~1まで。何点以上ならその人と判定するか
PassScore=0.9
#画像の読み込み
def get_file(dir_path):
    filenames = os.listdir(dir_path)
    return filenames




#メイン開始
if __name__ == "__main__":
    #画像を読み込んで、ファイル名をリスト化する。
    pic = get_file(testpic)
    print(pic)
    cnt=0 #正解数初期化
    #モデルの読み込み
    start_tj=t.time()
    model = model_from_json(open(keras_model).read())
    end_tj=t.time()


    start_tw=t.time()
    model.load_weights(keras_param)
    end_tw=t.time()
    #model.summary()
    with open("./model/categories.json",'r') as fi: 
        classes=json.load(fi)
        classes["?"]="Unknown"
        print(classes)
    ##ここまでで実行するとモデルの形が結果に表示される
    label_array=[]
    #リスト化したファイルから読み込んで処理する
    for i in pic:
        print(i) # ファイル名の出力
        
        #画像ディレクトリにあるファイルのi番目を読み込み
        img = load_img(testpic + i,target_size=(224,224))
        # 画像を要素に取る配列(images)にする必要がある
        images = np.array([np.array(img)])
        
        start_tp=t.time()
        prd = model.predict(images)
        end_tp=t.time()

        for j in range(people):
            print(classes[str(j)]+"の確率->"+"{0:3.3f}".format(prd[0][j]*100)+"%")
        
        #確信度最大値を取得する
        prelabel = prd.argmax(axis=1)
        if(prd.max()>PassScore):
            label=prelabel[0]
        else:
            label='?'
        label_array.append(label)
        #print([classes[c] for c in str(label)])
        print(classes[str(label)])
        if re.match(classes[str(label)], i):
            cnt+=1
            print("正解^-^")
        else:
            print("不正解-_-")
        print()
        print()
    backend.clear_session()
    end_t=t.time()
    print("*---結果---*")
    print("CKFaceの解答=>",str(label_array)) 
    print("[DEBUG]正解数" + str(cnt) + "/" + str(len(label_array)))
    print("[DEBUG]正答率"+str(cnt/len(label_array)*100)+"%")
    countLabel=collections.Counter(label_array)
    result=countLabel.most_common(1)
    # print("あなたは"+classes[str(result[0][0])]+"さんです。")

   
    
    print("【keras_auth.py】"+"{0:.5f}".format(end_t-start_t)+"秒")
    print("【model_from_json】"+"{0:.5f}".format(end_tj-start_tj)+"秒")
    print("【load_weights】"+"{0:.5f}".format(end_tw-start_tw)+"秒")
    print("【predict】"+"{0:.5f}".format(end_tp-start_tp)+"秒")
