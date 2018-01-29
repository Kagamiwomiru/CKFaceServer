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


