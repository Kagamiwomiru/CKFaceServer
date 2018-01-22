#!/bin/bash
mkdir result/DRP
for i in `seq 10`
do
	echo $i
	python3 face_keras.py >./result/DRP/Server$i.txt
	./exe_client.sh >./result/DRP/Client$i.txt
done
