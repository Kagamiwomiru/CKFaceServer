scp face/face-model.h5 Raspi:CKFace/face/face-model.h5
scp face/face.json Raspi:CKFace/face/face.json
scp face/categories.json Raspi:CKFace/face/categories.json
ssh Raspi 'cd CKFace/ ;python3 keras_auth.py'