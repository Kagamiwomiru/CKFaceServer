# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import AveragePooling2D
from keras.utils import np_utils
from keras.applications.xception import Xception
from keras.optimizers import SGD
from keras import callbacks
from keras.backend import tensorflow_backend as backend
import MakeData as DA
import json

# 画像サイズ．ResNetを使う時は224
img_size = 224
batch_size = 16
#以下ディレクトリに入っている画像を読み込む
root_dir = "./face/"
#学習データを何周するか
epochs=20
#ログファイル
log_filepath="./logs/"
#学習したモデル
ModelWeightData = "./model/face-model.h5"
ModelArcData = "./model/face.json"
classFile = "./model/categories.json"

#学習率(SGD(lr=???))
learning_rate=0.04
def main():
    mizumashi_generator,val_generator,valX,valy=data_augmentation()
    x,base_model=model_load()
    model,opt=model_build(mizumashi_generator,x,base_model)
    model=learning(model,opt,mizumashi_generator,val_generator)

    # class と indexの対応を逆にする
    indices_to_class = dict((v, k)   for k, v in mizumashi_generator.class_indices.items())
    print(indices_to_class)
    f=open(classFile,'w')
    json.dump(indices_to_class,f)
    model_eval(model, valX, valy)
    backend.clear_session()




def data_augmentation():
    #輝度をあげる
    os.system('bash ./initface.sh')
    DA.sepia('sp')
    DA.Shape(10, -1, 'spsp', '0')    #学習画像データを水増し（データ拡張）を行う
    mizumashi_data=ImageDataGenerator()
    mizumashi_generator=mizumashi_data.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)
    #テスト画像データを水増しする。
    val_datagen=ImageDataGenerator()
    val_gen=val_generator=val_datagen.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=False)
    valX,valy=val_gen.next()
    return (mizumashi_generator,val_generator,valX,valy)

def model_load():
    #重みvをimagenetとすると、学習済みパラメータを初期値としてXceptionを読み込む。
    base_model = Xception(weights='imagenet', include_top=False,
                         input_tensor=Input(shape=(img_size,img_size, 3)))
   #base_model.summary()
    x=base_model.output
    #入力を平滑化
    x=Flatten()(x)
    #過学習防止
    x=Dropout(.4)(x)

    return (x,base_model)


def model_build(mizumashi_generator,x,base_model):
    # 最後の全結合層の出力次元はクラスの数(= mizumashi_generator.num_class)
    predictions = Dense(mizumashi_generator.num_classes,kernel_initializer='glorot_uniform', activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    opt = SGD(lr=learning_rate)
    #opt = Adam()
    return (model,opt)

def learning(model,opt,mizumashi_generator,val_generator):
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    tb_cb=keras.callbacks.TensorBoard(log_dir=log_filepath,histogram_freq=0)
    es_cb=keras.callbacks.EarlyStopping(monitor='val_loss',patience=0,verbose=1,mode='auto')
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
    
    return model

def model_eval(model, X, y):
    score = model.evaluate(X, y,batch_size=batch_size)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
		main()

