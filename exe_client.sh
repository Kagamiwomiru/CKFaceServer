scp face/face-model.h5 Raspi:CKFace/face-model.h5
scp face/face.json Raspi:CKFace/face.json
scp face/categories.json Raspi:CKFace/categories.json
sshpass -p issi0505 ssh Raspi 'cd CKFace/ ;python3 keras_auth.py'