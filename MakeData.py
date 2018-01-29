
# coding: utf-8

# In[133]:


import cv2
import numpy as np
import sys
import os
import glob


# In[236]:


#ハイコントラスト画像を作成します。引数にはファイルの末尾につける水増し識別子を指定します。hoge_識別子.jpg
def high_cont(id):
    #ハイコントラストルックアップテーブル作成
    min_table = 0
    max_table = 100
    diff_table = max_table - min_table
    LUT_HC = np.arange(256, dtype='uint8')

    for i in range(0, min_table):
        LUT_HC[i] = 0

    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table

    for i in range(max_table, 255):
        LUT_HC[i] = 255
    root_dir='face/'

    root_dirs=os.listdir(root_dir)


    for img_dir in root_dirs:
        print(img_dir)
        img=glob.glob(root_dir+img_dir+'/*.jpg')
        trans_img = []
        for i in img:

            # 画像の読み込み
            img_src = cv2.imread(i,1)
            trans_img.append(img_src)
            #画像変換
            hight_cont_imgs=[]
            for i in trans_img:
                hight_cont_img=cv2.LUT(i,LUT_HC)
                hight_cont_imgs.append(hight_cont_img)
            # 保存
            cnt=0
            for i in hight_cont_imgs:
                write=root_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,i)
                cnt+=1
#エッジ抽出を行います。
def edge_detection(id):
    root_dir='face/'
    root_dirs=os.listdir(root_dir)

    for img_dir in root_dirs:
        print(img_dir)
        img=glob.glob(root_dir+img_dir+'/*.jpg')
        trans_img = []
        for i in img:

            # 画像の読み込み
            img_src = cv2.imread(i,1)
            trans_img.append(img_src)
            #画像変換
            edge_imgs=[]
            for i in trans_img:
                gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
                retval, binarized = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY_INV)
                edges = cv2.Canny(gray,50,150,apertureSize = 3)
                edge_imgs.append(edges)
            # 保存
            cnt=0
            for i in edge_imgs:
                write=root_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,i)
                cnt+=1

#学習データを削除します。
def data_eraser(ers_number):
    root_dir='face/'
    root_dirs=os.listdir(root_dir)
    for img_dir in root_dirs:
        img=glob.glob(root_dir+img_dir+'/*.jpg')
        for i in range(ers_number):
            os.remove(img[i])
