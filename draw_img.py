import glob
import os
import shutil
import numpy as np
from scipy.misc import toimage
import matplotlib.pyplot as plt
from matplotlib import gridspec
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
def draw_img(datagen):
#出力先ディレクトリを指定
    temp_dir="./result/tmpimg"
    result_images="./result/out.png"
    os.mkdir(temp_dir)

#generatorから9個の画像を生成
#xはサンプル数で考慮(1枚ならbatch_size=1)
    for i in range(9):
        batch=datagen.next()

    #生成した画像を3x3で描画
    images = glob.glob(os.path.join(temp_dir,"*.png"))
    fig=plt.figure()
    gs=gridspec.GridSpec(3,3)
    gs.update(wspace=0.1,hspace=0.1)
    for i in range(9):
        img=load_img(images[i])
        plt.subplot(gs[i])
        plt.imshow(img,aspect='auto')
        plt.axis("off")
        plt.savefig(result_images)
    shutil.rmtree(temp_dir)

