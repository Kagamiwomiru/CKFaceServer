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
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras import callbacks
from keras.backend import tensorflow_backend as backend

# 画像サイズ．ResNetを使う時は224
img_size = 224
batch_size = 8
#以下ディレクトリに入っている画像を読み込む
root_dir = "./face/"
#学習データを何周するか
epochs=10
#学習したモデル
ModelWeightData="./face/face-model.h5"
ModelArcData="./face/face.json"
NumpyFile="./face/face.npy"

def main():
    mizumashi_generator,val_generator=data_augmentation()
    x,base_model=load_model()
    model,opt=model_build(mizumashi_generator,x,base_model)
    model=learning(model,opt,mizumashi_generator,val_generator)

    X_test, y_test = np.load(NumpyFile)
    X_test  = X_test.astype("float")  / 256
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    
    model_eval(model, X_test, y_test)


def load_people():
    #categories = ["Kagami","Kato","Uchiyama","the_others"]
    categories=[]
    for x in os.listdir(root_dir):
     if os.path.isdir(root_dir + x):
         categories.append(x)
    print(categories)
    #リスト書き出し
    f = open("./categories.txt","w")
    for x in categories:
       f.write(str(x) + "\n")
    f.close()

    #people=3
    return len(categories)


def data_augmentation():
    #学習画像データを水増し（データ拡張）を行う
    mizumashi_data=ImageDataGenerator()
    mizumashi_generator=mizumashi_data.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=True)
    #テスト画像データを水増しする。
    val_datagen=ImageDataGenerator()
    val_generator=val_datagen.flow_from_directory(directory=root_dir,target_size=(img_size,img_size),batch_size=batch_size,shuffle=False)

    return (mizumashi_generator,val_generator)

def load_model():
    #重みをimagenetとすると、学習済みパラメータを初期値としてResNet50を読み込む。
    base_model = ResNet50(weights='imagenet', include_top=False,
                         input_tensor=Input(shape=(img_size, img_size, 3)))
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
    opt = SGD(lr=.01, momentum=.9)
    return (model,opt)

def learning(model,opt,mizumashi_generator,val_generator):
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit_generator(mizumashi_generator,
                                  validation_data=val_generator,
                                  steps_per_epoch=mizumashi_generator.samples // batch_size,
                                  validation_steps=val_generator.samples // batch_size,
                                  epochs=epochs,
                                  verbose=1)
    #モデルの構造と重みを保存。
    json_string=model.to_json()
    open(ModelArcData,'w').write(json_string)
    model.save_weights(ModelD)
    
    return model





"""
def build_model(in_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3,
	border_mode='same',
	input_shape=in_shape))
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
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',
	optimizer='rmsprop',
	metrics=['accuracy'])
    return model

def model_train(X, y):
    model = build_model(X.shape[1:])
    tf_lg=keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1)
    log=[tf_lg]
    history = model.fit(X, y, batch_size=32, nb_epoch=epochs,callbacks=log,validation_split=0.1)
    json_string=model.to_json()
    open('./face/face.json','w').write(json_string)
    hdf5_file = "./face/face-model.h5"
    model.save_weights(hdf5_file)
    return model
"""
def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
    backend.clear_session()
if __name__ == "__main__":
		nb_classes=load_people()
		main()

