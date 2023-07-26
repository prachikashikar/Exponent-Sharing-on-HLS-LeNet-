// The layers are modified to read weights from three different variables i.e, sign, index and mantissa. 
#include "LeNet.h"
#include "stdio.h"
#include "stdlib.h"

//Exponent table with distinct values
ap_int<8> wExpList [20] = {126, 124, 128, 125, 123, 122, 127, 120, 121, 119, 129, 118, 116, 117, 112, 115, 108, 113, 111, 114};

// Read Floats in IEEE format and seggregate sign, exponent and mantissa from that
union myFloat
{
	float f;
	struct
	{
		unsigned int mantissa:23;
		unsigned exponent:8;
		unsigned sign:1;
	}raw;
};

float expf(float x) {
 x = 1.0 + x / 1024;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x;
 return x;
}

float Conv_5x5(float input[25], float kernel[25]){
	int x,y;
	float result = 0;
	for(y = 0; y < 5; y++){
		for(x = 0; x < 5; x++){
			result += input[x+y*5] * kernel[x+y*5];
		}
	}
	return result;
}

//kernel 5x5x6 = 25x6 = 150
void ConvLayer_1(float input[1024],float * C1_value, ap_int<1> * wSign, ap_int<5> * wInd, ap_int<23> * wMant){
	int i_y,i_x,matrix_y,matrix_x;
	int k_num,mat_i = 0,input_value_index;
	myFloat temp;
	top_loop:for(int k_num = 0; k_num < 6; k_num+=1){
		//TODO memory kernel
		float matrix_2[25];
		for(mat_i = 0;mat_i<25;mat_i++){
			\\Retrieve the weights
			temp.raw.sign = wSign[mat_i + k_num*25];
			temp.raw.exponent = wExpList[wInd[mat_i + k_num*25]];
			temp.raw.mantissa = wMant[mat_i + k_num*25];
			matrix_2[mat_i] = temp.f;
		}
		i_y_loop:for(i_y = 0; i_y < 28; i_y++){
			for(i_x = 0; i_x < 28; i_x++){
				float matrix[25];
				int pic_value_index = i_x + i_y * 32;
				matrix_loop:for(matrix_y = 0; matrix_y <5; matrix_y++){
					caculate:for(matrix_x = 0; matrix_x <5; matrix_x++){
#pragma HLS pipeline II = 1
//						Image index 0 ~ 24
						int matrix_index = matrix_x + matrix_y * 5;
//						Image pixel index 0 ~ 1024, related to matrix_x, matrix_y, x, y=32
						input_value_index = pic_value_index + matrix_x + matrix_y * 32;
						matrix[matrix_index] = input[input_value_index];

					}
				}
				int out_pic_index = i_x + i_y * 28 + k_num * 784;
				C1_value[out_pic_index] = Conv_5x5(matrix,matrix_2);
			}
		}
	}
}

float AvgPool_2x2(float input[4]){
	float res = 0;
	int i;
	for(i = 0; i < 4 ; i++){
		res += input[i];
	}
	res /= 4;
	return res;
}

float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

void AvgpoolLayer_2(float input[4704],float *A2_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 6; k_num++){
		for(i_y = 0; i_y < 27; i_y+=2){
			for(i_x = 0;  i_x < 27; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 28 + k_num * 784;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 28 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A2_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}

//kernel 5x5x6x16 = 25x6x16 =2400
void ConvLayer_3(float input[1176],float *C3_value,ap_int<1> * wSign, ap_int<5> * wInd, ap_int<23> * wMant){
	int k_num,nk_num,i_y,i_x,matrix_x,matrix_y;
	int mat_i;
	myFloat temp;
    for(nk_num = 0; nk_num < 16; nk_num++){
		for(i_y = 0; i_y < 10; i_y++){
			for(i_x = 0; i_x < 10; i_x++){
				float res = 0;
				float res_total_6 = 0;
				float matrix[25];
				int index_now = i_x + i_y * 10 + nk_num * 100;
				for(k_num = 0; k_num < 6; k_num++){
					float matrix_2[25];
					for(mat_i = 0;mat_i<25;mat_i++){
						\\Retrieve the weights
						int weights_index = mat_i + k_num*25 + (nk_num+1)*150;
						temp.raw.sign = wSign[weights_index];
						temp.raw.exponent = wExpList[wInd[weights_index]];
						temp.raw.mantissa = wMant[weights_index];
						matrix_2[mat_i] = temp.f;
					}
					for(matrix_y = 0; matrix_y <5; matrix_y++){
						for(matrix_x = 0; matrix_x <5; matrix_x++){
#pragma HLS pipeline II = 1
							int matrix_index = matrix_x + matrix_y * 5;
							int input_value_index = index_now + matrix_x + matrix_y * 14;
							matrix[matrix_index] = input[input_value_index];
						}
					}
					res_total_6 += Conv_5x5(matrix,matrix_2);
				}
				C3_value[index_now] = res_total_6;
			}
		}
	}
}
void AvgpoolLayer_4(float input[1600],float *A4_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 16; k_num++){
		for(i_y = 0; i_y < 10; i_y+=2){
			for(i_x = 0;  i_x < 10; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 10 + k_num * 100;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 10 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A4_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}
//kernel  5x5x16x120 = 48000
void FullyConnLayer_5(float input[400],float *F5_value,ap_int<1> * wSign, ap_int<5> * wInd, ap_int<23> * wMant ){
	int i_y,i_x;
	myFloat temp;
	for(i_y = 0; i_y < 120; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 400; i_x++){
#pragma HLS pipeline II = 1
			int index = i_x + i_y * 400;
			int wt_ind = index + 2550;
			\\Retrieve the weights
			temp.raw.sign = wSign[wt_ind];
			temp.raw.exponent = wExpList[wInd[wt_ind]];
			temp.raw.mantissa = wMant[wt_ind];
			res += input[i_x] * temp.f;
		}
		F5_value[i_y] = res;
	}
}





//kernel 84x120 = 10080
void FullyConnLayer_6(float input[120],float *F6_value,ap_int<1> * wSign, ap_int<5> * wInd, ap_int<23> * wMant ){
	int i_y,i_x;
	myFloat temp;
	for(i_y = 0; i_y < 84; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 120; i_x++){
#pragma HLS pipeline II = 1
			int index = i_x + i_y * 120;
			int wt_ind = index + 50550;
			\\Retrieve the weights
			temp.raw.sign = wSign[wt_ind];
			temp.raw.exponent = wExpList[wInd[wt_ind]];
			temp.raw.mantissa = wMant[wt_ind];
			res += input[i_x] * temp.f;
		}
		F6_value[i_y] = res;
	}
}

int c = 0;
//kernel 10x84 = 840
void FullyConnLayer_7(float input[84],float *F6_value,ap_int<1> * wSign, ap_int<5> * wInd, ap_int<23> * wMant ){
	int i_y,i_x;
	myFloat temp;
	for(i_y = 0; i_y < 10; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 84; i_x++){
#pragma HLS pipeline II = 1
			int index = i_x + i_y * 84;
			c++;
			\\Retrieve the weights
			temp.raw.sign = wSign[index + 60630];
			temp.raw.exponent = wExpList[wInd[index + 60630]];
			temp.raw.mantissa = wMant[index + 60630];
			res += input[i_x] * temp.f;
		}
		F6_value[i_y] = res;
	}
}

int Softmax_1_8(float input[10],float *probability,float *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index;
}


void LetNet(ap_int<1> signW[62494],ap_int<5> indW[62494], ap_int<23> mantW[62494] , int* r){


	// 32x32 image
	float photo[1024];
	//layer1 weights  5x5x6 = 25x6 = 150
	//layer3 weights  5x5x6x16 = 25x6x16 =2400
	//layer5 weights 400x120 = 48000
	//layer6 weights 84x120 = 10080
	//layer7 weights 10x84 = 840

	//The output of each layer
	float C1_value[4704];
	float A2_value[1176];
	float C3_value[1600];
	float A4_value[400];
	float F5_value[120];
	float F6_value[84];
	float F7_value[10];

	float probability[10];
	float res[10];
	int loop1_i;


	//get the image data
	for(loop1_i = 0; loop1_i<1024; loop1_i++){
		int t = loop1_i;
		photo[loop1_i] = (float)t;
	}
	//calulation of each layer
	//Pass sign, index and mantissa to each layer other that passing a float as weight
	ConvLayer_1(photo,C1_value,signW,indW,mantW);
	AvgpoolLayer_2(C1_value,A2_value);
	ConvLayer_3(A2_value,C3_value,signW,indW,mantW);
	AvgpoolLayer_4(C3_value,A4_value);
	FullyConnLayer_5(A4_value,F5_value,signW,indW,mantW);
	FullyConnLayer_6(F5_value,F6_value,signW,indW,mantW);
	FullyConnLayer_7(F6_value,F7_value,signW,indW,mantW);
	*r = Softmax_1_8(F7_value,probability,res);
}
