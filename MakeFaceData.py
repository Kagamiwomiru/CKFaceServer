import os, glob
import numpy as np
from sklearn import cross_validation
from keras.preprocessing.image import load_img, img_to_array

root_dir = "./face/"
#categories=["Kagami","Kato","Uchiyama","the_others"]
categories=[]
for x in os.listdir(root_dir):
    if os.path.isdir(root_dir + x):
        categories.append(x)
print(categories)
nb_classes = len(categories)
image_size = 96

X = []
Y = []

for idx ,cat  in enumerate(categories):
    files = glob.glob(root_dir +"/" + cat + "/*")
    print("~~~",cat,"を処理中")
    for i, f in enumerate(files):
        img = load_img(f, target_size=(image_size,image_size))
        data = img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save("./face/face.npy", xy)
print("ok,", len(Y))
