import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import AveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
from keras import callbacks

import numpy as np

# 画像サイズ．ResNetを使う時は224
img_size = 224
batch_size = 8
# 学習データを何周するか
epochs = 15

# train_dir 以下の画像を読み込む
train_dir = './face/'

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
    directory=train_dir, target_size=(img_size, img_size), batch_size=batch_size, shuffle=True)

valid_dir = './face/'

valid_datagen = ImageDataGenerator()
valid_generator = valid_datagen.flow_from_directory(
    directory=valid_dir, target_size=(img_size, img_size), batch_size=batch_size, shuffle=False)

# weights='imagenet' とすると学習済みパラメータを初期値としてResNet50に読み込む
base_model = ResNet50(weights='imagenet', include_top=False,
                         input_tensor=Input(shape=(img_size, img_size, 3)))
x = base_model.output
x = Flatten()(x)
x = Dropout(.4)(x)
# 最後の全結合層の出力次元はクラスの数(= train_generator.num_class)
predictions = Dense(train_generator.num_classes,
                    kernel_initializer='glorot_uniform', activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)


opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit_generator(train_generator,
                              validation_data=valid_generator,
                              steps_per_epoch=train_generator.samples // batch_size,
                              validation_steps=valid_generator.samples // batch_size,
                              epochs=epochs,
                              verbose=1)



