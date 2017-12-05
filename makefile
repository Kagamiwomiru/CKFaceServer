# コンパイラを指定
CC :=g++ -std=c++11 
# インクルードファイル等
CFLAGS :=`pkg-config opencv --cflags` `pkg-config opencv --libs`

converter : main.o 
	$(CC)  main.o -o converter   $(CFLAGS)

main.o : CKFaceServerKit.h



clean:
	$(RM) *.o
	$(RM) convert
