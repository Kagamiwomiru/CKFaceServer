# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.backend import tensorflow_backend as backend
root_dir = "./face/"
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
nb_classes = len(categories)
image_size = 96
epochs=10
def main():
    X_train, X_test, y_train, y_test = np.load("./face/face.npy")
    X_train = X_train.astype("float") / 256
    X_test  = X_test.astype("float")  / 256
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test  = np_utils.to_categorical(y_test, nb_classes)
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)
    #os.system("bash ./sendToClient.sh")

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

def model_eval(model, X, y):
    score = model.evaluate(X, y)
    print('loss=', score[0])
    print('accuracy=', score[1])
    backend.clear_session()
if __name__ == "__main__":
    main()
