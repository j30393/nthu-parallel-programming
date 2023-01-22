#include <cstdio>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cstring>
#include <iostream>
#include <algorithm>
bool changed = false;
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
void count_off_all(int& offset , int& allocated , int rank , int new_size , int n){
	int gap = n / new_size; 
	offset = rank * gap ;
	if(rank != new_size - 1){ // not the last one 
		allocated = gap;
	}
	else{
		allocated = n - gap * (new_size - 1);
	}
}

void get_min(float* arr_target , float* arr_source , int arr_target_len , int arr_source_len , int size , int rank){
	int cursor_1 = 0;
	int cursor_2 = 0 , cnt = 0;
	float* temp_arr = (float*) malloc(sizeof(float) *(size + 2));
	while(cnt < size){
		if(cursor_1 < arr_target_len && cursor_2 < arr_source_len){
			if(arr_target[cursor_1] < arr_source[cursor_2]){
				temp_arr[cnt++] = arr_target[cursor_1++];
			}
			else{
				temp_arr[cnt++] = arr_source[cursor_2++];
			}
		}
		else if(cursor_1 <= arr_target_len ){
			temp_arr[cnt++] = arr_target[cursor_1++];
		}
		else{
			temp_arr[cnt++] = arr_source[cursor_2++];
		}
	}
	/*for(int i = 0 ; i < arr_target_len ; i++){
		printf("get min function is called by %d and we have the value with target arr %f \n" , rank , arr_target[i]);
	}
	for(int i = 0 ; i < arr_source_len ; i++){
		printf("get min function is called by %d and we have the value with source arr %f\n" , rank , arr_source[i]);
	}*/
	for(int i = 0 ; i < size ; i++){
		// printf("%f ", temp_arr[i]);
		arr_target[i] = temp_arr[i];
	}
	if(cursor_1 != arr_target_len) changed = true;
	// printf("\n");
	free(temp_arr);
}

void get_max(float* arr_target , float* arr_source , int arr_target_len , int arr_source_len , int size , int rank){
	int cursor_1 = arr_target_len - 1;
	int cursor_2 = arr_source_len - 1 , cnt = 0;
	float* temp_arr = (float*) malloc(sizeof(float) *(size + 2));
	while(cnt < size){
		if(cursor_1 >= 0 && cursor_2 >= 0){
			if(arr_target[cursor_1] > arr_source[cursor_2]){
				temp_arr[cnt++] = arr_target[cursor_1--];
			}
			else{
				temp_arr[cnt++] = arr_source[cursor_2--];
			}
		}
		else if(cursor_1 >= 0 ){
			temp_arr[cnt++] = arr_target[cursor_1--];
		}
		else{
			temp_arr[cnt++] = arr_source[cursor_2--];
		}
	}
	/*for(int i = 0 ; i < arr_target_len ; i++){
		printf("get max function is called by %d and we have the value with target arr %f\n" , rank , arr_target[i]);
	}
	for(int i = 0 ; i < arr_source_len ; i++){
		printf("get max function is called by %d and we have the value with source arr %f \n" , rank , arr_source[i]);
	}*/
	for(int i = 0 ; i < size ; i++){
		//printf("%f ", temp_arr[i]);
		arr_target[size - i - 1] = temp_arr[i];
	}
	//printf("\n");
	if(cursor_1 != -1 ) changed = true;
	free(temp_arr);
}

int main(int argc, char** argv) {
	// n -> the size of array 
	int n = atoi(argv[1]);
	int rank, size;
	int new_size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm new_comm ; 
	// printf("rank is %d  and size is %d" , size , rank);
	if(size > n ){
		MPI_Group orig_group, new_group;
		MPI_Comm_group(MPI_COMM_WORLD, &orig_group);
		// printf("%d into the excl func \n", rank);
		// discard the extra 
		// Remove all unnecessary ranks
		int ranges[3] = { n  , size - 1 , 1 };
		MPI_Group_range_excl(orig_group, 1, &ranges, &new_group);
		bool flag = (rank >= n) ;
		// Create a new communicator
		MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
		if(new_comm == MPI_COMM_NULL){
			if(flag){
				// printf("abort the rank  %d \n " , rank);
				MPI_Finalize();
				return 0;
			}
		}
	}
	else{
		new_comm = MPI_COMM_WORLD;
	}
	MPI_Comm_size(new_comm, &new_size);
	// printf("We have the new size %d \n" , new_size);

	int offset , allocated_job_size; 
	count_off_all(offset , allocated_job_size , rank , new_size , n);

	// printf("the %d node has the offset of %d and with the allocated job size %d \n" , rank , offset , allocated_job_size );
	
	float* local_data = (float*) malloc(sizeof(float) * (allocated_job_size + size)); // + new_size -> make sure it have enough space
	MPI_File input_file , output_file;

	MPI_File_open(new_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
	MPI_File_read_at(input_file, sizeof(float) * offset, local_data, allocated_job_size, MPI_FLOAT, MPI_STATUS_IGNORE);
	
	/*for(int i = 0 ; i < allocated_job_size ; i++){
		printf("rank %d got float: %f\n", rank, local_data[i]);
	}*/
	radix_sort(local_data, allocated_job_size); // do the first radix sort of local data

	int odd_rank,even_rank;
	/* avoid touching illegal ranks */
	int odd_off , odd_allocate ;
	int even_off , even_allocate;

    if(rank%2==0){
		odd_rank = rank - 1; 
		even_rank = rank + 1;
	}
	else {
		odd_rank = rank + 1;
		even_rank = rank - 1;
	}

	if (odd_rank == -1 || odd_rank == new_size ){
		odd_rank = MPI_PROC_NULL;
		odd_allocate = allocated_job_size;
	} 
	else{count_off_all(odd_off , odd_allocate , odd_rank , new_size , n);}
	if (even_rank == -1 || even_rank == new_size ){
		even_rank = MPI_PROC_NULL;
		even_allocate = allocated_job_size;
	} 
	else{count_off_all(even_off , even_allocate , even_rank , new_size , n);}
	float* recv_data_even = (float*) malloc(sizeof(float) * (even_allocate + size));
	float* recv_data_odd = (float*) malloc(sizeof(float) * (odd_allocate + size));
	bool all_changed = false; 
	for (int p = 0; p < new_size * 2  ; p++) {

		if(p%2 == 0){ // even phase
			if(even_rank != MPI_PROC_NULL){
				//printf("this is rank %d and p is %d , it's even phase , with the  %d being rival \n" , rank , p , even_rank);
				// printf("the rank %d it's going to send %d bits of data and receive %d bits of data  from %d \n" , rank, allocated_job_size , even_allocate, even_rank );
				MPI_Sendrecv(local_data, allocated_job_size, MPI_FLOAT, even_rank, 0, recv_data_even, 
				even_allocate,  MPI_FLOAT ,even_rank, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE); 
				if(rank % 2 == 0){
					get_min(local_data , recv_data_even , allocated_job_size , even_allocate, allocated_job_size , rank);
				}
				else{
					get_max(local_data , recv_data_even , allocated_job_size , even_allocate, allocated_job_size , rank);
				}
			}
			else{
				/*MPI_Sendrecv(local_data, allocated_job_size , MPI_FLOAT, rank, 1, recv_data_even,
				allocated_job_size, MPI_FLOAT, rank, 1, new_comm, MPI_STATUS_IGNORE);
				printf("this is rank %d and p is %d , it's even phase , with the  %d being rival \n" , rank , p , even_rank);
				printf("the rank %d it's going to send %d bits of data and receive %d bits of data  from %d \n" , rank, allocated_job_size , even_allocate, even_rank );
				for(int i = 0 ; i < allocated_job_size ; i++){
					printf("rank %d have float: %f\n", rank, local_data[i]);
				}
				printf("The phase is even and I'm rank %d and I don't need no exchange" , rank );*/
			}
		}
		else{
			if(odd_rank != MPI_PROC_NULL){
				//printf("this is rank %d and p is %d , it's odd phase , with the  %d being rival \n" , rank , p , odd_rank);
				//printf("the rank %d it's going to send %d bits of data and receive %d bits of data  from %d \n" , rank, allocated_job_size , odd_allocate, odd_rank );
				MPI_Sendrecv(local_data, allocated_job_size, MPI_FLOAT, odd_rank, 0, recv_data_odd, 
				odd_allocate,  MPI_FLOAT ,odd_rank, MPI_ANY_TAG, new_comm, MPI_STATUS_IGNORE); 
				if(rank % 2 == 0){
					get_max(local_data , recv_data_odd , allocated_job_size , odd_allocate, allocated_job_size , rank);
				}
				else{
					get_min(local_data , recv_data_odd , allocated_job_size , odd_allocate, allocated_job_size , rank);
				}
			}
			else{
				/*MPI_Sendrecv(local_data, allocated_job_size , MPI_FLOAT, rank, 1, recv_data_odd,
				allocated_job_size, MPI_FLOAT, rank, 1, new_comm, MPI_STATUS_IGNORE);*/
				/*printf("this is rank %d and p is %d , it's odd phase , with the  %d being rival \n" , rank , p , odd_rank);
				printf("the rank %d it's going to send %d bits of data and receive %d bits of data  from %d \n" , rank, allocated_job_size , even_allocate, even_rank );*/
				/*for(int i = 0 ; i < allocated_job_size ; i++){
					printf("rank %d have float: %f\n", rank, local_data[i]);
				}*/
				// printf("The phase is even and I'm rank %d and I don't need no exchange" , rank );
			}	
		}
		MPI_Allreduce(&changed, &all_changed, 1, MPI_CXX_BOOL, MPI_LOR, new_comm);
		if(p % 2 == 1 ){
			if(!all_changed){
				break;
			}
		}
		// MPI_Barrier(new_comm);
	}

	/*for(int i = 0 ; i < allocated_job_size ; i++){
		printf("rank %d got float: %f\n", rank, local_data[i]);
	}*/

	free(recv_data_even);
	free(recv_data_odd);
	MPI_File_open(new_comm, argv[3], MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
	MPI_File_write_at(output_file, offset * sizeof(float), local_data, allocated_job_size, MPI_FLOAT, MPI_STATUS_IGNORE);
	free(local_data);
	MPI_Finalize();
	
	return 0;
}
