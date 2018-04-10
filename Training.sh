#!/bin/bash
# python3 MakeFaceData.py
python3 face_keras.py
scp model/face-model.h5 Raspi:CKFace/model/face-model.h5
scp model/face.json Raspi:CKFace/model/face.json
scp model/categories.json Raspi:CKFace/model/categories.json
# ssh Raspi 'cd CKFace/ ;python3 keras_auth.py'
