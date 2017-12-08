#!/bin/bash
python3 MakeFaceData.py
python3 face_keras.py
scp face/face-model.h5 Raspi:CKFace/face-model.h5
scp face/face.json Raspi:CKFace/face.json

