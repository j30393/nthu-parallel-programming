/*
templated on the github 
modified by Peter Su 2023/1/4
*/
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda.h>

#define NUM_ITERATION 10000
#define INIT_TEMPERATURE 0.4
#define MIN_TEMPERATURE 0.001
#define INIT_TOLERANCE 1
#define DELTA_T 0.2
#define SODUKU_SIZE 9
#define Z_LAYER 50 
// modify: use multi-threads to try different random step
// Everything related to Z_LAYER is new. I won't label all of them.

__constant__ int d_mask[SODUKU_SIZE*SODUKU_SIZE];
char outname[50];
//Error Checks

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel for initializing random number generators
__global__ void init_random_generator(curandState *state) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(1337, idx, 0, &state[idx]);
}

// This functions returns the count of number of unique elements in a row or column number according to the flag (Device Version)
__device__ int d_num_unique(int rc_num,int sudoku[][SODUKU_SIZE][SODUKU_SIZE],int flag, int z_idx)
{
	// Z_LAYER modify: int sudoku[SODUKU_SIZE][SODUKU_SIZE] -> int sudoku[Z_LAYER][SODUKU_SIZE][SODUKU_SIZE]
	int nums[SODUKU_SIZE]={1,2,3,4,5,6,7,8,9};
	int idx, unique_Count;

        unique_Count = 0;

        for(int j=0;j<SODUKU_SIZE;j++)
        {
            if(flag==2)
                idx = sudoku[z_idx][j][rc_num]-1;
            else
                idx = sudoku[z_idx][rc_num][j]-1;
            if(idx==-1)
                return -1;
            if(nums[idx]!=0)
            {
                unique_Count+=1;
                nums[idx]=0;
            }
        }

        return unique_Count;
}

//Computes the energy by adding the number of unique elements in all the rows and columns
__device__ int d_compute_energy(int sudoku[][SODUKU_SIZE][SODUKU_SIZE], int z_idx)
{
	int energy=0;

	for(int i=0;i<SODUKU_SIZE;i++)
        energy += d_num_unique(i,sudoku,1, z_idx) + d_num_unique(i,sudoku,2 ,z_idx);

	return 162-energy;
}

//Kernel to run a Markov chain
__global__ void markov(int* sudoku,curandState *state,int cur_energy,float temperature, \
	int *b1,int *b2,int *b3,int *b4,int *b5,int *b6,int *b7, \
	int *b8,int *b9,int *b10,int *b11,int *b12,int *b13,int *b14,int *b15,int *energy_block)
{
	__shared__ int shd_sudoku[Z_LAYER][SODUKU_SIZE][SODUKU_SIZE]; 
	
	int local_energy = cur_energy, local_cur_energy = cur_energy; 
	__shared__ int shared_local_energy[Z_LAYER]; // modify: for the root can choose from them later.

	int thread_x=threadIdx.x, thread_y=threadIdx.y;
	int thread_num_local= threadIdx.x*blockDim.x + threadIdx.y;
	int block_num= blockIdx.x*blockDim.x + blockIdx.y;
	int state_idx = SODUKU_SIZE*SODUKU_SIZE*blockIdx.y + threadIdx.x*SODUKU_SIZE + threadIdx.y;

	// Bring the sudoku to shared memory
	for(int i = 0 ; i < Z_LAYER ; i++){
		shd_sudoku[i][thread_x][thread_y]=sudoku[thread_x+ SODUKU_SIZE*thread_y];
	}
	__syncthreads();

	// if(thread_num_local!=0) return;
	if(thread_num_local >= Z_LAYER) return;

	// index of randomly seleted pairs
	int block_x, block_y;
	int r1_x, r1_y, r2_x, r2_y;
	int temp;

	for(int iter=0;iter<NUM_ITERATION;iter++)
	{
		// Select a random sub grid
		// modify: change &state[block_num] to &state[state_idx] to get randomness among threads
		block_x = 3*(int)(3.0*curand_uniform(&state[state_idx]));
		block_y = 3*(int)(3.0*curand_uniform(&state[state_idx]));

		// Select two random mutable points ( ie. 0 in given sudoku ) randomly
		do
		{
			r1_x=(int)3.0*curand_uniform(&state[state_idx]);
			r1_y=(int)3.0*curand_uniform(&state[state_idx]);
		}while(d_mask[(block_x+r1_x)+SODUKU_SIZE*(block_y+r1_y)]==1);


		do{
			r2_x=(int)3.0*curand_uniform(&state[state_idx]);
			r2_y=(int)3.0*curand_uniform(&state[state_idx]);
		}while(d_mask[(block_x+r2_x)+SODUKU_SIZE*(block_y+r2_y)]==1);

		// Swap the values of these randomly selected points 
		temp=shd_sudoku[thread_num_local][block_x+r1_x][block_y+r1_y];
		shd_sudoku[thread_num_local][block_x+r1_x][block_y+r1_y]=shd_sudoku[thread_num_local][block_x+r2_x][block_y+r2_y];
		shd_sudoku[thread_num_local][block_x+r2_x][block_y+r2_y]=temp;

		// Compute the energy of this new state
		local_energy=d_compute_energy(shd_sudoku, thread_num_local);

		// Accept the new state if its energy is lower than the previous state
		if (local_energy < local_cur_energy) local_cur_energy = local_energy;

		// Else still accept or reject the new state with the acceptance probability
		else
		{
			// Accept the state
			if (exp((float)(local_cur_energy - local_energy) / temperature) > curand_uniform(&state[state_idx])) local_cur_energy = local_energy;

			// Reject the state and undo changes
			else
			{
				temp = shd_sudoku[thread_num_local][block_x + r1_x][block_y + r1_y];
				shd_sudoku[thread_num_local][block_x + r1_x][block_y + r1_y] = shd_sudoku[thread_num_local][block_x + r2_x][block_y + r2_y];
				shd_sudoku[thread_num_local][block_x + r2_x][block_y + r2_y] = temp;
			}
		}

		//If reached the lowest point break
		if(local_energy==0) break;
	}
	shared_local_energy[thread_num_local] = local_energy;

	__syncthreads();

	// printf("block %d  current energy = %d\n", block_num, cur_energy);

	// Choose the best one among all threads
	int min = 200, min_idx = 200;
	if(thread_num_local != 0) return;
	for(int i = 0 ; i < Z_LAYER ; i++)
	{
		if(shared_local_energy[i] < min)
		{
			min = shared_local_energy[i];
			min_idx = i;
		}
	}

	// **note**
	// we can let the all the threads write back to original ones , but when the size goes larger , we're no longer avaible 
	// So the way I have done is that only let on thread to write the whole sudoku back 

	//Write the result back to memory
	for(int i=0;i<SODUKU_SIZE;i++)
	{
		for(int j=0;j<SODUKU_SIZE;j++)
		{
			if(block_num==0)
				b1[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==1)
				b2[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==2)
				b3[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==3)
				b4[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==4)
				b5[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==5)
				b6[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==6)
				b7[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==7)
				b8[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==8)
				b9[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==9)
				b10[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==10)
				b11[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==11)
				b12[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==12)
				b13[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==13)
				b14[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
			if(block_num==14)
				b15[i+SODUKU_SIZE*j]=shd_sudoku[min_idx][i][j];
		}
	}

	//Write the energy back to memory for the current state
	energy_block[block_num]=min;
}

//Display the sudoku
void display_sudoku(int *n){

    printf("\n_________________________\n");
    for(int i=0;i<SODUKU_SIZE;i++){
        printf("| ");
        for(int j=0;j<SODUKU_SIZE;j=j+3)
            printf("%1d %1d %1d | ",n[i+SODUKU_SIZE*j],n[i+SODUKU_SIZE*(j+1)],n[i+SODUKU_SIZE*(j+2)]);
        if((i+1)%3==0){
            printf("\n-------------------------\n");
        }else printf("\n");
    }
    return;
}

/*Initialize the sudoku. 1) Read the partial sudoku.
					     2) Place values in all the empty slots such that the 3x3 subgrid clause is satisfied */
void init_sudoku(int *s,int *m,char* fname)
{
	FILE *fin ;
	fin = fopen(fname,"r");

	//Output file name
	int len;
	for(len=0;len<strlen(fname)-2;len++)
		outname[len]=fname[len];
	strcat(outname,"out");

	int in;

	int x, y;
	int p, q;
	int idx;

	int nums_1[SODUKU_SIZE],nums_2[SODUKU_SIZE];


	//Read the partial sudoku from file
	//Compute the mask. 0 -> mutable value 1-> non-mutable

	for(int i=0;i<SODUKU_SIZE;i++){

		for(int j=0;j<SODUKU_SIZE;j++){

			fscanf(fin,"%1d",&in);
			s[i+SODUKU_SIZE*j] = in;
			if(in==0)
				m[i+SODUKU_SIZE*j]=0;
			else
				m[i+SODUKU_SIZE*j]=1;
			}
		}
	fclose(fin);

	printf("Puzzle\n");
	display_sudoku(s);

	//Place values in all the empty slots such that the 3x3 subgrid clause is satisfied
	for(int block_i=0;block_i<3;block_i++)
	{
		for(int block_j=0;block_j<3;block_j++)
		{
			for(int k=0;k<SODUKU_SIZE;k++)
				nums_1[k]=k+1;

				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;

						if(s[x+SODUKU_SIZE*y]!=0){
							p = s[x+SODUKU_SIZE*y];
							nums_1[p-1]=0;
						}
					}
				}
				q = -1;
				for(int k=0;k<SODUKU_SIZE;k++)
				{
					if(nums_1[k]!=0)
					{
						q+=1;
						nums_2[q] = nums_1[k];
					}
				}
				idx = 0;
				for(int i=0;i<3;i++)
				{
					for(int j=0;j<3;j++)
					{
						x = block_i*3 + i;
						y = block_j*3 + j;
						if(s[x+SODUKU_SIZE*y]==0)
						{
							s[x+SODUKU_SIZE*y] = nums_2[idx];
							idx+=1;
						}
					}
				}

			}
		}
}

// This functions returns the count of number of unique elements in a row or column number according to the flag (Host Version)
int h_num_unique(int i, int k, int *n){

    int nums[SODUKU_SIZE]={1,2,3,4,5,6,7,8,9};
    int idx, unique_count;

    unique_count = 0;

    for(int j=0;j<SODUKU_SIZE;j++){

        if(k==1){
            idx = n[i+SODUKU_SIZE*j]-1;
        }

        else{
            idx = n[j+SODUKU_SIZE*i]-1;
        }

        if(idx==-1){
            return -1;
        }

        if(nums[idx]!=0){
            unique_count+=1;
            nums[idx]=0;
        }
    }
    return unique_count;
}

//Computes the energy by adding the number of unique elements in all the rows and columns
int h_compute_energy(int *n)
{
	    int energy = 0;

	    for(int i=0;i<SODUKU_SIZE;i++){
	        energy += h_num_unique(i,1,n) + h_num_unique(i,2,n);
	    }

	    return 162 - energy;
}

void write_file(int *s)
{
	FILE *fout;
	fout=fopen(outname,"w");

	for(int i=0;i<SODUKU_SIZE;i++)
	{
		for(int j=0;j<SODUKU_SIZE;j++)
			fprintf(fout,"%1d",s[i+SODUKU_SIZE*j]);
		if(i<8)
		fprintf(fout,"\n");
	}

	fclose(fout);

}

//Main
int main(int arg,char* argv[]) {
	//cudaSetDevice(0);
	//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,device);

	//Tunable Parameter
	int num_chains;
	if(prop.multiProcessorCount>=15)
		num_chains=15;
	else
		num_chains=prop.multiProcessorCount;
	
	float temperature=INIT_TEMPERATURE;
	float temp_min=MIN_TEMPERATURE;

	//Host pointers
	int *sudoku;
	int *mask;
	int *h_energy_host;

	int size=sizeof(int)*SODUKU_SIZE*SODUKU_SIZE;

	//Allocate memory
	gpuErrchk(cudaHostAlloc((void**)&sudoku,size,cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&mask,size,cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void**)&h_energy_host,sizeof(int)*num_chains,cudaHostAllocDefault));
	init_sudoku(sudoku,mask,argv[1]);

	//Initial Energy of sudoku
	int current_energy=h_compute_energy(sudoku);
	printf("Initial energy %d \n",current_energy);

	//Device pointers
	int *d_sudoku;
	int *d_b1,*d_b2,*d_b3,*d_b4,*d_b5,*d_b6,*d_b7,*d_b8,*d_b9,*d_b10,*d_b11,*d_b12,*d_b13,*d_b14,*d_b15;
	int *energy_block; // store the energy for num_chain blocks

	//Allocate memory
	gpuErrchk(cudaMalloc((void**)&d_sudoku,size));
	gpuErrchk(cudaMalloc((void**)&d_mask,size));
	gpuErrchk(cudaMalloc((void**)&d_b1,size));
	gpuErrchk(cudaMalloc((void**)&d_b2,size));
	gpuErrchk(cudaMalloc((void**)&d_b3,size));
	gpuErrchk(cudaMalloc((void**)&d_b4,size));
	gpuErrchk(cudaMalloc((void**)&d_b5,size));
	gpuErrchk(cudaMalloc((void**)&d_b6,size));
	gpuErrchk(cudaMalloc((void**)&d_b7,size));
	gpuErrchk(cudaMalloc((void**)&d_b8,size));
	gpuErrchk(cudaMalloc((void**)&d_b9,size));
	gpuErrchk(cudaMalloc((void**)&d_b10,size));
	gpuErrchk(cudaMalloc((void**)&d_b11,size));
	gpuErrchk(cudaMalloc((void**)&d_b12,size));
	gpuErrchk(cudaMalloc((void**)&d_b13,size));
	gpuErrchk(cudaMalloc((void**)&d_b14,size));
	gpuErrchk(cudaMalloc((void**)&d_b15,size));
	gpuErrchk(cudaMalloc((void**)&energy_block,sizeof(int)*num_chains));

	//Copy Sudoku and Mask to GPU
	gpuErrchk(cudaMemcpy(d_sudoku,sudoku,size,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpyToSymbol(d_mask,mask,size));

	//Grid and Block dimensions
	dim3 dimGrid(1,num_chains);
	dim3 dimBlock(SODUKU_SIZE,SODUKU_SIZE);

	//Random number generators. Launch init_random_generator kernel
	curandState *d_state;
	gpuErrchk(cudaMalloc(&d_state, dimBlock.x* dimBlock.y * dimGrid.x * dimGrid.y));
	init_random_generator<<<dimGrid.x * dimGrid.y, dimBlock.x* dimBlock.y>>>(d_state);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());


	int tolerance=INIT_TOLERANCE;
	int min,min_idx, e; // min energy and its block index
	int prev_energy=current_energy;

	
	//Simulated Annealing loop
	int cnt=0;
	do{
		min=200;
		min_idx=200;
		markov<<< dimGrid,dimBlock >>>(d_sudoku,d_state,current_energy,temperature,d_b1,d_b2,d_b3,d_b4,d_b5,d_b6,d_b7,d_b8,d_b9,d_b10,d_b11,d_b12,d_b13,d_b14,d_b15,energy_block);
		// gpuErrchk(cudaDeviceSynchronize());

		gpuErrchk(cudaMemcpy(h_energy_host,energy_block,sizeof(int)*num_chains,cudaMemcpyDeviceToHost));

		for(e=0;e<num_chains;e++)
		{
			if(h_energy_host[e]<min)
			{
				min=h_energy_host[e];
				min_idx=e;
			}

		}
		printf("Loop %d, block %d get min energy = %d \n ",cnt++, min_idx , min);

		if(min_idx==0)
		{
			gpuErrchk(cudaMemcpy(d_sudoku,d_b1,size,cudaMemcpyDeviceToDevice));
			current_energy=min;
		}

		if(min_idx==1)
		{
			cudaMemcpy(d_sudoku,d_b2,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==2)
		{
			cudaMemcpy(d_sudoku,d_b3,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==3)
		{
			cudaMemcpy(d_sudoku,d_b4,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==4)
		{
			cudaMemcpy(d_sudoku,d_b5,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==5)
		{
			cudaMemcpy(d_sudoku,d_b6,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==6)
		{
			cudaMemcpy(d_sudoku,d_b7,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==7)
		{
			cudaMemcpy(d_sudoku,d_b8,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==8)
		{
			cudaMemcpy(d_sudoku,d_b9,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==9)
		{
			cudaMemcpy(d_sudoku,d_b10,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==10)
		{
			cudaMemcpy(d_sudoku,d_b11,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==11)
		{
			cudaMemcpy(d_sudoku,d_b12,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==12)
		{
			cudaMemcpy(d_sudoku,d_b13,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==13)
		{
			cudaMemcpy(d_sudoku,d_b14,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(min_idx==14)
		{
			cudaMemcpy(d_sudoku,d_b15,size,cudaMemcpyDeviceToDevice);
			current_energy=min;
		}

		if(current_energy==0) break;

		// Random restart if energy is stuck
		if(current_energy==prev_energy) tolerance--;
		else tolerance=INIT_TOLERANCE;
		
		if(tolerance<0)
		{
			//printf("Randomizing\n");
			gpuErrchk(cudaMemcpy(sudoku,d_sudoku,size,cudaMemcpyDeviceToHost));

			int ar[3]={0,3,6};
				int tempa;
				int rand1=random()%3;
				int rand2=random()%3;

				int r1_x,r1_y,r2_x,r2_y;
				int block_x,block_y;

			for(int suf=0;suf<random()%10;suf++)
			{
				block_x = ar[rand1];
				block_y = ar[rand2];
				do{
					r1_x=random()%3;
					r1_y=random()%3;
				}while(mask[(block_x+r1_x)+SODUKU_SIZE*(block_y+r1_y)]==1);

				do{
					r2_x=random()%3;
					r2_y=random()%3;
				}while(mask[(block_x+r2_x)+SODUKU_SIZE*(block_y+r2_y)]==1);

				tempa=sudoku[(block_x+r1_x)+SODUKU_SIZE*(block_y+r1_y)];
				sudoku[(block_x+r1_x)+SODUKU_SIZE*(block_y+r1_y)]=sudoku[(block_x+r2_x)+SODUKU_SIZE*(block_y+r2_y)];
				sudoku[(block_x+r2_x)+SODUKU_SIZE*(block_y+r2_y)]=tempa;
			}
			gpuErrchk(cudaMemcpy(d_sudoku,sudoku,size,cudaMemcpyHostToDevice));
			current_energy=h_compute_energy(sudoku);
			//printf("Energy after randomizing %d \n",current_energy);
			tolerance=INIT_TOLERANCE;
			temperature=temperature+DELTA_T;
		}

		prev_energy=current_energy;

		if(current_energy==0)
		{
			break;
		}
		// modify: the decay to 0.99 which is suggested by the paper 
		temperature=temperature*0.8;

		//printf("Energy after temp %f is %d \n",temperature,current_energy);
	}while(temperature>temp_min);


	gpuErrchk(cudaMemcpy(sudoku,d_sudoku,size,cudaMemcpyDeviceToHost));

	display_sudoku(sudoku);

	write_file(sudoku);

	current_energy=h_compute_energy(sudoku);

	printf("Final energy %d \n",current_energy);

	return 0;
}
