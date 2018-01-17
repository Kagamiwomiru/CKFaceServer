#!/bin/bash
#モデルの比較実験を行います。

for i in `seq 1`
do

python3 face_keras_MDL/face_keras_Xce.py >./result/CKFaceServer/Xce$i.txt
./exe_client.sh >./result/CKFace/Xce$i.txt
python3 face_keras_MDL/face_keras_VGG16.py >./result/CKFaceServer/VGG16$i.txt
./exe_client.sh >./result/CKFace/VGG16$i.txt
python3 face_keras_MDL/face_keras_VGG19.py >./result/CKFaceServer/VGG19$i.txt
./exe_client.sh >./result/CKFace/VGG19$i.txt
python3 face_keras_MDL/face_keras_RES.py >./result/CKFaceServer/RES$i.txt
./exe_client.sh >./result/CKFace/RES$i.txt
python3 face_keras_MDL/face_keras_incV3.py >./result/CKFaceServer/incV3$i.txt
./exe_client.sh >./result/CKFace/incV3$i.txt
python3 face_keras_MDL/face_keras_InRes.py >./result/CKFaceServer/InRes$i.txt
./exe_client.sh >./result/CKFace/InRes$i.txt
python3 face_keras_MDL/face_keras_mobile.py >./result/CKFaceServer/mobile$i.txt
./exe_client.sh >./result/CKFace/mobile$i.txt
python3 face_keras_MDL/face_keras_Nas_L.py >./result/CKFaceServer/Nas_L$i.txt
./exe_client.sh >./result/CKFace/Nas_L$i.txt
python3 face_keras_MDL/face_keras_Nas_M.py >./result/CKFaceServer/Nas_M$i.txt
./exe_client.sh >./result/CKFace/Nas_M$i.txt
done