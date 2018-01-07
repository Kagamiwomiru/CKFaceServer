#!/bin/bash
#最適化関数を総当たり実行します。条件はgooglekeep参照
python3 face_keras_SGD.py >./result/CKFaceServer/SGD.txt
./exe_client.sh >./result/CKFace/SGD.txt
python3 face_keras_RMSprop.py >./result/CKFaceServer/RMSprop.txt
./exe_client.sh >./result/CKFace/RMSprop.txt
python3 face_keras_Adagrad.py >./result/CKFaceServer/Adagrad.txt
./exe_client.sh >./result/CKFace/Adagrad.txt
python3 face_keras_Adadelta.py >./result/CKFaceServer/Adadelta.txt
./exe_client.sh >./result/CKFace/Adadelta.txt
python3 face_keras_Adam.py >./result/CKFaceServer/Adam.txt
./exe_client.sh >./result/CKFace/Adam.txt
python3 face_keras_Adamax.py >./result/CKFaceServer/Adamax.txt
./exe_client.sh >./result/CKFace/Adamax.txt
python3 face_keras_Nadam.py >./result/CKFaceServer/Nadam.txt
./exe_client.sh >./result/CKFace/Nadam.txt
