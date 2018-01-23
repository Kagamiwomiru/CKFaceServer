#!/bin/bash
#水増し処理を比較します
for i in `seq 1`
do
    python3 face_keras_Zoom_near.py zoomNear $i > result/zoomNear/Server$i.txt
    ./exe_client.sh > result/zoomNear/Client$i.txt
    python3 face_keras_Zoom_wrap.py zoomWrap $i > result/zoomWrap/Server$i.txt
    ./exe_client.sh > result/zoomWrap/Client$i.txt 
    python3 face_keras_chanel.py chanel $i > result/chanel/Server$i.txt
    ./exe_client.sh > result/chanel/Client$i.txt
    python3 face_keras_shear.py shear $i > result/shear/Server$i.txt
    ./exe_client.sh > result/shear/Client$i.txt 
    python3 face_keras_Seikika.py Seikika $i > result/Seikika/Server$i.txt
    ./exe_client.sh > result/Seikika/Client$i.txt
done
    
    