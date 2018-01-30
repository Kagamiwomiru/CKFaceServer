
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import sys
import os
import glob


# In[236]:


#ハイコントラスト画像を作成します。引数にはファイルの末尾につける水増し識別子を指定します。hoge_識別子.jpg
def high_cont(id,flag):
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
    
    if(flag=='0'):
        root_dir='face/'
    else:
        root_dir='face.bak/'
        
    out_dir='face/'

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
                write=out_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,i)
                cnt+=1


# In[137]:


# エッジ抽出を行います
def edge_detection(id):
    root_dir='face.bak/'
    out_dir='face/'
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
                write=out_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,i)
                cnt+=1


# In[9]:


# データを指定数削除します。
def data_eraser(ers_number):
    root_dir='face/'
    root_dirs=os.listdir(root_dir)
    for img_dir in root_dirs:
        img=sorted(glob.glob(root_dir+img_dir+'/*.jpg'))
        for i in range(ers_number):
            os.remove(img[i])


# In[37]:


get_ipython().run_cell_magic('bash', '', './initface.sh')


# In[13]:

#画像をセピアにします。
def sepia(id):
    root_dir='face.bak/'
    out_dir='face/'
    root_dirs=os.listdir(root_dir)
    for img_dir in root_dirs:
            print(img_dir)
            img=glob.glob(root_dir+img_dir+'/*.jpg')
            trans_img = []
            for i in img:
                # 画像の読み込み
                img_src = cv2.imread(i)
                trans_img.append(img_src)
    #             セピア変換
            sepia_imgs=[]
            for im in trans_img:
                b, g, r = im[:,:,0],im[:,:,1], im[:,:,2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b 
                r=gray*240/255
                g=gray*200/255
                b=gray*145/255
                im[:,:,0],im[:,:,1], im[:,:,2]=b, g, r
                sepia_imgs.append(im)
             # 保存
            cnt=0

            for a in sepia_imgs:
                write=out_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,a)
                cnt+=1


# In[36]:

#条件付きグレースケール化します。　{'red,'green,'blue'},id
def CE_gray(color,id):
    root_dir='face.bak/'
    out_dir='face/'
    root_dirs=os.listdir(root_dir)
    for img_dir in root_dirs:
                print(img_dir)
                img=glob.glob(root_dir+img_dir+'/*.jpg')
                trans_img = []
                for i in img:
                     # 画像の読み込み
                    img_src = cv2.imread(i)
                    trans_img.append(img_src)
                # 条件付きグレースケール変換
                gray_imgs=[]
                for im in trans_img:
                    b, g, r = im[:,:,0],im[:,:,1], im[:,:,2]
                    if(color=='red'):
                        gray = r
                    elif(color=='green'):
                        gray=g
                    elif(color=='blue'):
                        gray=b

                    gray_imgs.append(gray)
                #保存
                cnt=0
                for a in gray_imgs:
                    write=out_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                    cv2.imwrite(write,a)
                    cnt+=1


# 画像をシャープにします。変化を加えた画像に対してシャープにしたい場合、第４引数を'0'にします。
def Shape(k,bit,id,flag):#k=シャープの度合い(10がオススメ）bit=ビット深度（OpenCVのが使えます。-1がオススメ),id,flag=root_dirから読み込む(0)
    
    if(flag=='0'):
        root_dir='face/'
    else:
        root_dir='face.bak/'
       
    out_dir='face/'
    root_dirs=os.listdir(root_dir)
#     id='rw'
#     #ビット深度
#     bit=-1
#     # シャープの度合い
#     k = 10.0
    # シャープ化するためのオペレータ
    shape_operator = np.array([[0,-k, 0],
                  [-k, 1 + 4 * k, -k],
                  [0, -k, 0]])
    for img_dir in root_dirs:
        print(img_dir)
        img=glob.glob(root_dir+img_dir+'/*.jpg')
        trans_img = []
        for i in img:
             # 画像の読み込み
            img_src = cv2.imread(i)
            trans_img.append(img_src)
            shape_imgs=[]
            for im in trans_img:
                img_tmp = cv2.filter2D(im, bit, shape_operator)
                img_shape = cv2.convertScaleAbs(img_tmp)
                shape_imgs.append(img_shape)
            #保存
            cnt=0
            for a in shape_imgs:
                write=out_dir+img_dir+'/'+str(cnt)+'_'+str(id)+'.jpg'
                cv2.imwrite(write,a)
                cnt+=1