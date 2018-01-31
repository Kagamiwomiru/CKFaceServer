#!/bin/bash
#水増し処理を比較します
echo [START TIME] `date`
mkdir result/AL/AL5
mkdir result/AL/AL6
mkdir result/AL/AL7
mkdir result/AL/AL8
mkdir result/AL/AL9
mkdir result/AL/AL10
mkdir result/AL/AL11
for i in `seq 20`
do
    python3 AugmentationLearning.py 0 > result/AL/AL5/$i.txt
    python3 AugmentationLearning.py 1 > result/AL/AL6/$i.txt
    python3 AugmentationLearning.py 2 > result/AL/AL7/$i.txt
    python3 AugmentationLearning.py 3 > result/AL/AL8/$i.txt
    python3 AugmentationLearning.py 4 > result/AL/AL9/$i.txt
    python3 AugmentationLearning.py 5 > result/AL/AL10/$i.txt
    python3 AugmentationLearning.py 6 > result/AL/AL11/$i.txt
done
    
echo [END TIME] `date`