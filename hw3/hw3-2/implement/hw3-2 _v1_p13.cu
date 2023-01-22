#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BlockSize 32

const int INF = ((1 << 30) - 1);
/* start some declaration */

int* Dist; 
int n , m , new_n;
/* end the declaration */

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void phase1(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize][BlockSize];
    int x_real = threadIdx.x + round_cnt * BlockSize;
    int y_real = threadIdx.y + round_cnt * BlockSize;
    int x = threadIdx.x;
    int y = threadIdx.y;

    /*if(x_real >= new_n || y_real  >= new_n){
        return;
    }*/
    store[x][y] = Dist_d[x_real * new_n + y_real];
    
    __syncthreads();
    for(int i = 0 ; i < BlockSize ; i++){
        store[x][y] = min(store[x][i] + store[i][y] , store[x][y]);
        __syncthreads();
    }
    Dist_d[x_real * new_n + y_real] = store[x][y];  
}

__global__ void phase2(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize][BlockSize];
    __shared__ int vertical[BlockSize][BlockSize];
    __shared__ int herizonal[BlockSize][BlockSize];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int block_num = blockIdx.y;
    int x_ver = threadIdx.x + round_cnt * BlockSize;
    int x_her = threadIdx.x + block_num * BlockSize;
    int y_ver = threadIdx.y + block_num * BlockSize;
    int y_her = threadIdx.y + round_cnt * BlockSize;
    // if duplicate with the phase 1
    if(block_num == round_cnt ){
        return;
    }
    // since we calculate both vertical and herizonal at once ,we can't delete both given one doesn't fulfill
    store[x][y] = Dist_d[x_ver * new_n + y_her];
    herizonal[x][y] = Dist_d[ x_her * new_n + y_her];
    vertical[x][y] = Dist_d[ x_ver * new_n + y_ver ];
    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        vertical[x][y] = min( store[x][i] + vertical[i][y] , vertical[x][y]);
        herizonal[x][y] =  min( herizonal[x][i] + store[i][y] , herizonal[x][y]);
    }
    Dist_d[x_her * new_n + y_her] = herizonal[x][y];
    Dist_d[x_ver * new_n + y_ver] = vertical[x][y];
}

__global__ void phase3(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize][BlockSize];
    __shared__ int vertical[BlockSize][BlockSize];
    __shared__ int herizonal[BlockSize][BlockSize];
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;
    if(block_x == round_cnt || block_y == round_cnt){
        return;
    }
    int x_real = block_x * BlockSize + x;
    int y_real = block_y * BlockSize + y;
    int x_her = x_real;
    int y_her = round_cnt * BlockSize + y;
    int x_ver = round_cnt * BlockSize + x;
    int y_ver = y_real;
    if(x_real >= new_n || y_real >= new_n){
        return;
    }
    store[x][y] = Dist_d[x_real * new_n + y_real];
    vertical[x][y] = Dist_d[x_ver * new_n + y_ver];
    herizonal[x][y] = Dist_d[x_her * new_n + y_her];
    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[x][y] = min(herizonal[x][i] + vertical[i][y] , store[x][y]);
    }
    Dist_d[x_real * new_n + y_real] = store[x][y] ;
}

void input(char* infile) {

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    new_n = ceil(n , BlockSize) * BlockSize; // for the purpose to make coalsememory
    Dist = (int*) malloc(new_n * new_n *sizeof(int));

    for (int i = 0; i < new_n; ++i) {
        for (int j = 0; j < new_n ; ++j) {
            if (i == j) {
                Dist[i * new_n + j] = 0;
            } else {
                Dist[i * new_n + j] = INF;
            }
        }
    }


    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * new_n + pair[1]] = pair[2];
    }

    /*for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            printf("%d " , Dist[i * new_n + j]);
        }
        printf("\n");
    }*/
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
		fwrite(Dist + i * new_n , sizeof(int), n , outfile);
	}
    for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            printf("%d " , Dist[i * new_n + j]);
        }
        printf("\n");
    }
    fclose(outfile);
}



int main(int argc, char* argv[]) {
    input(argv[1]);
    // int BlockSize = 512;
    int* deviceDist;
    cudaMalloc(&deviceDist, new_n * new_n * sizeof(int));
	cudaMemcpy(deviceDist, Dist, new_n * new_n * sizeof(int), cudaMemcpyHostToDevice);

    int total_block_num = new_n / BlockSize;
    dim3 block_num1(1, 1);
    dim3 block_num2(1, total_block_num);
    dim3 block_num3(total_block_num, total_block_num);
    dim3 thread_each_block(BlockSize,BlockSize);
    // printf("the total block : %d \n" , total_block_num);
    // printf("the n is %d and the new_n is %d \n" ,  n , new_n);
    
    for(int i = 0 ; i < total_block_num ; i++){
        phase1<<<block_num1 , thread_each_block>>>(deviceDist , i , new_n); // phase 1 
        phase2<<<block_num2 , thread_each_block>>>(deviceDist , i , new_n); // phase 2
        phase3<<<block_num3 , thread_each_block>>>(deviceDist , i , new_n); // phase 3
    }
    cudaMemcpy(Dist , deviceDist, new_n * new_n * sizeof(int), cudaMemcpyDeviceToHost); 
    output(argv[2]);
    return 0;
}