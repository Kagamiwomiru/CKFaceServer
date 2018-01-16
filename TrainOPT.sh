#!/bin/bash
#最適化関数を総当たり10回実行します。条件はgooglekeep参照
for i in `seq 10`
do

python3 face_keras_SGD.py >./result/CKFaceServer/SGD$i.txt
./exe_client.sh >./result/CKFace/SGD$i.txt
python3 face_keras_RMSprop.py >./result/CKFaceServer/RMSprop$i.txt
./exe_client.sh >./result/CKFace/RMSprop$i.txt
python3 face_keras_Adagrad.py >./result/CKFaceServer/Adagrad$i.txt
./exe_client.sh >./result/CKFace/Adagrad$i.txt
python3 face_keras_Adadelta.py >./result/CKFaceServer/Adadelta$i.txt
./exe_client.sh >./result/CKFace/Adadelta$i.txt
python3 face_keras_Adam.py >./result/CKFaceServer/Adam$i.txt
./exe_client.sh >./result/CKFace/Adam$i.txt
python3 face_keras_Adamax.py >./result/CKFaceServer/Adamax$i.txt
./exe_client.sh >./result/CKFace/Adamax$i.txt
python3 face_keras_Nadam.py >./result/CKFaceServer/Nadam$i.txt
./exe_client.sh >./result/CKFace/Nadam$i.txt
done