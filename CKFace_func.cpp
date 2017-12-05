#include"CKFaceKit.h"
void cap_image(Mat &in,string path, int width, int height,int num_images,VideoCapture cap) {
	int count=0;
	//VideoCapture cap(0);
	VideoWriter writer(path, 0, 0.0, Size(width, height));
	while(count<=num_images){	
		cap >> in;
		namedWindow("CKFace_Capturing",0);
		setWindowProperty("CKFace_Capturing",CV_WND_PROP_FULLSCREEN,CV_WINDOW_FULLSCREEN);
		imshow("CKFace_Capturing",in);
		writer << in;
		count++;
		waitKey(30);
		}
	destroyWindow("CKFace_Capturing");
	//cap.release();
}


void TrainMode(Mat frame,string input_path,string output_path,int *PICT,VideoCapture cap,int *time_convert,int *time_cap){
	char Name[NAME_SIZE];//名前
	char sendcmd[SYSCMD];//画像をサーバに送る
	char execmd[SYSCMD];//サーバ処理を実行
	char svrm[SYSCMD];//サーバ側の./face/Nameを削除
	char rmcmd[SYSCMD];//クライアント側の"./face/*を削除
	char mkcmd[SYSCMD];//クライアント側の./face/にNameのディレクトリ作成
	char out_path[255];
	clock_t start_convert,end_convert;//コンバート時間
//	clock_t start_send,end_send;
	clock_t start_cap,end_cap;//cap時間
	destroyWindow("CKFace_View");
	printf("**Welcome to TrainMode!!**\n");
	printf("input your Name\n");
	printf("=>");
	scanf("%s",Name);
	printf("Hello %s!\n",Name);
	sprintf(out_path,"./face/%s",Name);
	sprintf(mkcmd,"mkdir %s",out_path);
	strcat(out_path,"/%1d");
	output_path=string(out_path);
	system(mkcmd);
	start_cap=clock();
	cap_image(frame,output_path, WIDTH,HEIGHT,*PICT,cap);
	end_cap=clock();
		/*printf("**Converting...**\n");
	printf("この処理にはしばらく時間がかかります。\n");
	start_convert=clock();
	convert_image(input_path,output_path,*PICT);
	end_convert=clock();
	*/
	sprintf(sendcmd,"scp -r face/%s Neptune:CKFaceServer/face/",Name);
	sprintf(rmcmd,"rm -rf face/%s",Name);
	sprintf(svrm,"ssh Neptune rm -rf CKFaceServer/face/%s",Name);
	sprintf(execmd,"ssh Neptune 'cd CKFaceServer/;./Training.sh'");
	printf("**Learning...**\n");
	printf("サーバ側の「face/%s」ディレクトリを初期化しています...\n",Name);
	system(svrm);
	printf("サーバに送信しています...\n");
	//start_send=clock();
	system(sendcmd);
	//end_send=clock();
	printf("ラズパイの「face/%s」ディレクトリを削除しています...\n",Name);
	system(rmcmd);
	printf("学習中です。しばらくお待ちください。\n");
	system(execmd);
	*time_cap=end_cap-start_cap;
	*time_convert=end_convert-start_convert;
	//*time_send=end_send-start_send;
	return ;
}


void AuthMode(Mat frame,string input_path,string output_path,int *PICT,VideoCapture cap){
	printf("**Welcome to AuthMode!!**\n");
	destroyWindow("CKFace_View");
	*PICT=15;
	cap_image(frame,input_path, WIDTH,HEIGHT,*PICT,cap);
	convert_image(input_path,output_path,*PICT);
	printf("D:ここから認証処理\n");
	return;
}

void ManualMode(Mat frame,string input_path,string output_path,int *PICT,VideoCapture cap,int *time_convert,int *time_cap){
	int mode=2;
	printf("**Welcome to ManualMode!**\n");
	printf("PICT:");
	scanf("%d",PICT);
	printf("What are you doing?\n");
	printf("0=TrainMode,1=AuthMode,2=Quit\n");
	printf("=>");
	scanf("%d",&mode);
		if(mode==0) TrainMode(frame,input_path,output_path,PICT,cap,time_convert,time_cap);
		else if(mode==1) AuthMode(frame,input_path,output_path,PICT,cap);
	
	return ;
}

