#include <cstdio>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

static inline unsigned int FloatFlip(float input){
	unsigned int f = *(unsigned int *)&input;
	unsigned int mask = -int(f >> 31) | 0x80000000; // -1 = 0xFFFFFFFF
	return f ^ mask;
}

static inline float IFloatFlip(unsigned int f){ // reverse back
	unsigned int mask = ((f >> 31) - 1) | 0x80000000;
	unsigned ret = mask ^ f;
	return *(float *)&(ret);
}

void radix_sort(float* input_data,int size){

	unsigned int* arr = (unsigned int*)malloc((size + 5) * sizeof(unsigned int));
	unsigned int* sort = (unsigned int*)malloc((size +5 ) * sizeof(unsigned int));
	
	// 3 histograms on the stack:
	const int kHist = 65536;
	unsigned int b0[kHist * 2];

	unsigned int *b1 = b0 + kHist;

	memset(b0, 0, sizeof(unsigned int)* kHist * 2 );

	// set the whole number of 
	for (int i = 0; i < size; i++) {
		unsigned int fi = FloatFlip(input_data[i]);
		arr[i] = fi;
		b0[fi & 0xFFFF] ++; // first 16 bits
		b1[fi >> 16 ] ++; // 12~22 bits
		// printf("%d " , fi);
	}
	
	
	{
		unsigned int sum0 = 0, sum1 = 0;	
		for (int i = 0; i < kHist; i++) {
			unsigned int tsum;
			tsum = b0[i] + sum0;
			b0[i] = sum0;
			sum0 = tsum;
			tsum = b1[i] + sum1;
			b1[i] = sum1;
			sum1 = tsum;
		}
	}

	//  countSort for bit 0
	for (int i = 0; i < size ; i++) {
		unsigned int fi = arr[i];
		unsigned int pos = fi & 0xFFFF;
		sort[b0[pos]++] = fi;
	}

	// countSort for bit 1
	for (int i = 0; i < size; i++) {
		unsigned int fi = sort[i];
		unsigned int pos = (fi >> 16);
		arr[b1[pos]++] = fi;
	}

	// to write original:
	for(int i = 0 ; i < size ; i++){
		input_data[i] = IFloatFlip(arr[i]);
	}
	
	free(arr);
	free(sort);
}

void get_max(float* arr_target , float* arr_source , int arr_target_len , int arr_source_len , int size){
	int cursor_1 = arr_target_len - 1;
	int cursor_2 = arr_source_len - 1 , cnt = 0;
	float* temp_arr = (float*) malloc(sizeof(float) *size);
	while(cnt < size){
		if(cursor_1 > 0 && cursor_2 > 0){
			if(arr_target[cursor_1] > arr_source[cursor_2]){
				temp_arr[cnt++] = arr_target[cursor_1--];
			}
			else{
				temp_arr[cnt++] = arr_source[cursor_2--];
			}
		}
		else if(cursor_1 > 0 ){
			temp_arr[cnt++] = arr_target[cursor_1--];
		}
		else{
			temp_arr[cnt++] = arr_source[cursor_2--];
		}
	}
	for(int i = 0 ; i < size ; i++){
		arr_target[i] = temp_arr[i];
	}
	free(temp_arr);
}

int main(int argc, char** argv) {
	float* arr_1 = (float*) malloc(sizeof(float) *5);
    float* arr_2 = (float*) malloc(sizeof(float) *4);
    arr_1[0] = 500;
    arr_1[1] = -0.2;
    arr_1[2] = -100000;
    arr_1[3] = 201212312315;
    arr_1[4] = 5001231521212315;
    arr_2[0] = -20;
    arr_2[1] = -10;
    arr_2[2] = -5;
    arr_2[3] = -1;
	radix_sort(arr_1 , 5);
    // get_max(arr_1 , arr_2 , 5 , 4 , 5);
    for(int i = 0 ; i < 5 ; i++){
        std::cout << arr_1[i] << " ";
    }
    /*for(int i = 0 ; i < 4 ; i++){
        std::cout << arr_2[i] << " ";
    }
	std::cout << IFloatFlip(FloatFlip(-17513.746094));*/
}
