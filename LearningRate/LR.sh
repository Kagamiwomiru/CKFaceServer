#!/bin/bash
#learning rate for SGDの比較をします。
echo [START TIME] `date`

for cnt in `seq 10`
do
i=1
    for j in `seq 9`
    do

    python3  face_keras_LR.py  0.0$i $cnt   > ../result/LR/0.0$i/Server$cnt.txt
    i=$(( i + 1 ))
    done
    python3  face_keras_LR.py  0.1 $cnt   > ../result/LR/0.1/Server$cnt.txt
    
done

echo [END TIME] `date`
