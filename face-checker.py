import face_keras as face
import sys, os, glob
from PIL import Image
import numpy as np
import time
from keras.backend import tensorflow_backend as backend

A = time.time()
root_dir = "./face/"
image_size = 96
#categories = ["Kagami", "Kato","Uchiyama","the_others"]
categories=[]
for x in os.listdir(root_dir):
    if os.path.isdir(root_dir + x):
        categories.append(x)
print(categories)

image_dir ="./test_face"
#files = glob.glob(image_dir + "/*.jpg")
files = glob.glob(image_dir + "/*.jpg")


# 入力画像をNumpyに変換 --- (※2)
X = []
change_files = []
for fname in files:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    change_files.append(fname)
X = np.array(X)
X  = X.astype("float")  / 256
# CNNのモデルを構築 --- (※3)
model = face.build_model(X.shape[1:])
model.load_weights(root_dir + "face-model.h5")
# データを予測 --- (※4)
predict=model.predict(X)
for pre in predict:
    y=pre.argmax()
    print("D:categories=",categories[y])
"""
pre = model.predict(X)
y=0

personal = 0
nonper = 0
for i, p in enumerate(pre):
    personal=personal+p[0]
    nonper=nonper+p[1]
    
    print("+ 入力:", files[i])
    print(pre[i])
    if (pre[0][y] >= 0.8):
        break
    else:
        y+=1
personal=personal/len(pre)
nonper=nonper/len(pre)
print(personal)

"""

print("◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆◇◆")
#print("判定結果：",max(personal,nonper)*100,"％の確率で",categories[y],"です")
if(y<14):
    print("判定結果：",categories[y],"です")
else:
    print("まことに申し訳ございませんが、誰の顔なのかわかりませんでした。")


B = time.time()
print('time=',format(B-A))

backend.clear_session()
