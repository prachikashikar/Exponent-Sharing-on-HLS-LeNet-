#include "LeNet.h"
#include "stdlib.h"
#include <time.h>
#include "fstream"
#include "ap_int.h"
using namespace std;
int main(){
	srand((unsigned)time(NULL));
	ap_int<1> wSign[62494];
	ap_int<5> wInd[62494];
	ap_int<23> wMant[62494];
	int r = 0,t;
	ifstream wSignFile,wIndFile,wMantFile;
	wSignFile.open("Wsigns.txt");
	wIndFile.open("Windices.txt");
	wMantFile.open("Wmantissa.txt");

//	float test[1024]={0},weights[51750]={0};
	for(int i_2 = 0; i_2<61830; i_2++){
		wSignFile >> t;
		wSign[i_2] = t;
		wIndFile >> t;
		wInd[i_2] = t;
		wMantFile >> t;
		wMant[i_2] = t;
	}

	LetNet(wSign,wInd,wMant,&r);
	std::cout<<"res:"<<r<<"\n";

	wSignFile.close();
	wIndFile.close();
	wMantFile.close();

	return 0;
}
