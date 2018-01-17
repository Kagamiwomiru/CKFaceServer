#!/bin/bash
#モデルの比較実験を行います。
mkdir OUTPUT_MODEL/Xce & mkdir OUTPUT_MODEL/VGG16 & mkdir OUTPUT_MODEL/VGG19 & mkdir OUTPUT_MODEL/RES & mkdir OUTPUT_MODEL/incV3 & mkdir OUTPUT_MODEL/InRes & mkdir OUTPUT_MODEL/mobile & mkdir OUTPUT_MODEL/Nas_L & mkdir OUTPUT_MODEL/Nas_M 
for i in `seq 7`
do

python3 face_keras_MDL/face_keras_Xce.py Xce $i >./result/CKFaceServer/Xce$i.txt 
# ./exe_client.sh >./result/CKFace/Xce$i.txt &
python3 face_keras_MDL/face_keras_VGG16.py VGG16 $i >./result/CKFaceServer/VGG16$i.txt
# ./exe_client.sh >./result/CKFace/VGG16$i.txt &
python3 face_keras_MDL/face_keras_VGG19.py VGG19 $i >./result/CKFaceServer/VGG19$i.txt
# ./exe_client.sh >./result/CKFace/VGG19$i.txt &
python3 face_keras_MDL/face_keras_RES.py RES $i >./result/CKFaceServer/RES$i.txt
# ./exe_client.sh >./result/CKFace/RES$i.txt & 
python3 face_keras_MDL/face_keras_incV3.py incV3 $i >./result/CKFaceServer/incV3$i.txt
# ./exe_client.sh >./result/CKFace/incV3$i.txt &
python3 face_keras_MDL/face_keras_InRes.py InRes $i >./result/CKFaceServer/InRes$i.txt
# ./exe_client.sh >./result/CKFace/InRes$i.txt &
python3 face_keras_MDL/face_keras_mobile.py mobile $i >./result/CKFaceServer/mobile$i.txt
# ./exe_client.sh >./result/CKFace/mobile$i.txt &
python3 face_keras_MDL/face_keras_Nas_L.py Nas_L $i>./result/CKFaceServer/Nas_L$i.txt
# ./exe_client.sh >./result/CKFace/Nas_L$i.txt &
python3 face_keras_MDL/face_keras_Nas_M.py Nas_M $i>./result/CKFaceServer/Nas_M$i.txt
# ./exe_client.sh >./result/CKFace/Nas_M$i.txt
done