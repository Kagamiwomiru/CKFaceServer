#!/bin/bash
#水増し処理を比較します
mkdir result/AL/
for i in `seq 10`
do
    python3 AugmentationLearning.py 0 > result/AL/AL0$i.txt
    python3 AugmentationLearning.py 1 > result/AL/AL1$i.txt
    python3 AugmentationLearning.py 2 > result/AL/AL2$i.txt
    python3 AugmentationLearning.py 3 > result/AL/AL3$i.txt

done
    
    