#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BlockSize 64
#define Half 32
const int INF = ((1 << 30) - 1);
/* start some declaration */

int* Dist; 
int n , m , new_n;
/* end the declaration */

int ceil(int a, int b) { return (a + b - 1) / b; }

__device__ void autogenerate(int* Dist_d , int* target , int x , int y , int x_d  , int y_d , int new_n){

    target[y * BlockSize + x  ] = Dist_d[y_d * new_n + x_d];
    target[y * BlockSize + x + 1 ] = Dist_d[y_d * new_n + x_d + 1];
    target[y * BlockSize + x + 32] = Dist_d[y_d * new_n + x_d + 32];
    target[y * BlockSize + x + 33 ] = Dist_d[y_d * new_n + x_d + 33];
}

__device__ void write_back(int* Dist_d , int* target , int x , int y , int x_d  , int y_d , int new_n){
    Dist_d[y_d * new_n + x_d] = target[y * BlockSize + x ];
    Dist_d[y_d * new_n + x_d + 1] = target[y * BlockSize + x + 1];
    Dist_d[y_d * new_n + x_d + 32] = target[y * BlockSize + x + 32];
    Dist_d[y_d * new_n + x_d + 33] = target[y * BlockSize + x + 33];
}

__global__ void phase1(int* Dist_d, int round_cnt, int new_n){
    // input shape (16,64)
    __shared__ int store[BlockSize * BlockSize];
    int x = threadIdx.x * 2;
    int y = threadIdx.y;
    int x_real = x + round_cnt * BlockSize;
    int y_real = threadIdx.y + round_cnt * BlockSize;
    autogenerate(Dist_d ,store ,x ,y, x_real , y_real , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[y * BlockSize + x] = min(store[y * BlockSize + i] + store[i* BlockSize + x] , store[y * BlockSize + x]);
        store[y * BlockSize + x + 1] = min(store[y * BlockSize + i ] + store[i* BlockSize + x + 1] , store[y * BlockSize + x + 1]);
        store[y * BlockSize + x + 32] = min(store[y * BlockSize + i ] + store[i* BlockSize + x + 32] , store[y * BlockSize + x + 32]);
        store[y * BlockSize + x + 33] = min(store[y * BlockSize + i ] + store[i* BlockSize + x + 33] , store[y * BlockSize + x + 33]);
        __syncthreads();
    }
    write_back(Dist_d , store ,x ,y, x_real , y_real , new_n);
}

__global__ void phase2(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize * BlockSize];
    __shared__ int vertical[BlockSize * BlockSize];
    __shared__ int herizonal[BlockSize * BlockSize];
    int x = threadIdx.x * 2;
    int y = threadIdx.y;
    int block_num = blockIdx.y;
    
    int x_ver = x + round_cnt * BlockSize;
    int x_her = x + block_num * BlockSize;
    int y_ver = y + block_num * BlockSize;
    int y_her = y + round_cnt * BlockSize;
    
    // if duplicate with the phase 1
    /*if(block_num == round_cnt ){
        return;
    }*/
    // since we calculate both vertical and herizonal at once ,we can't delete both given one doesn't fulfill
    autogenerate(Dist_d ,store , x , y , x_ver , y_her , new_n);
    autogenerate(Dist_d ,herizonal , x , y , x_her , y_her , new_n);
    autogenerate(Dist_d ,vertical , x , y , x_ver , y_ver , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        vertical[y * BlockSize + x] = min( vertical[y * BlockSize + i] + store[i * BlockSize + x] , vertical[y * BlockSize + x]);
        vertical[y * BlockSize + x + 1] = min( vertical[y * BlockSize + i] + store[i * BlockSize + x + 1] , vertical[y * BlockSize + x + 1]);
        vertical[y * BlockSize + x + 32] = min( vertical[y * BlockSize + i] + store[i * BlockSize + x + 32] , vertical[y * BlockSize + x + 32]);
        vertical[y * BlockSize + x + 33] = min( vertical[y * BlockSize + i] + store[i * BlockSize + x + 33] , vertical[y * BlockSize + x + 33]);

        herizonal[y * BlockSize + x] =  min( store[y * BlockSize + i] + herizonal[i * BlockSize + x] , herizonal[y * BlockSize + x]);
        herizonal[y * BlockSize + x + 1] =  min( store[y * BlockSize + i] + herizonal[i * BlockSize + x + 1] , herizonal[y * BlockSize + x + 1]);
        herizonal[y * BlockSize + x + 32] =  min( store[y * BlockSize + i] + herizonal[i * BlockSize + x + 32] , herizonal[y * BlockSize + x + 32]);
        herizonal[y * BlockSize + x + 33] =  min( store[y * BlockSize + i] + herizonal[i * BlockSize + x + 33] , herizonal[y * BlockSize + x + 33]);
    }

    write_back(Dist_d ,herizonal , x , y , x_her , y_her , new_n);
    write_back(Dist_d ,vertical , x , y , x_ver , y_ver , new_n);
}

__global__ void phase3(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize * BlockSize];
    __shared__ int vertical[BlockSize * BlockSize];
    __shared__ int herizonal[BlockSize * BlockSize];
    int x = threadIdx.x * 2;
    int y = threadIdx.y;
    /*if(block_x == round_cnt || block_y == round_cnt){
        return;
    }*/
    int x_her = blockIdx.x * BlockSize + x;
    int y_her = round_cnt * BlockSize + y;
    int x_ver = round_cnt * BlockSize + x;
    int y_ver =  blockIdx.y * BlockSize + y;
    /*if(x_real >= new_n || y_real >= new_n){
        return;
    }*/
    autogenerate(Dist_d , store , x , y , x_her , y_ver , new_n);
    autogenerate(Dist_d , herizonal ,x , y, x_her , y_her , new_n);
    autogenerate(Dist_d , vertical , x , y ,x_ver , y_ver , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[y * BlockSize + x] = min(vertical[y * BlockSize + i] + herizonal[i * BlockSize + x] , store[y * BlockSize + x]);
        store[y * BlockSize + x + 1] = min(vertical[y * BlockSize + i] + herizonal[i * BlockSize + x + 1] , store[y * BlockSize + x + 1]);
        store[y * BlockSize + x + 32] = min(vertical[y * BlockSize + i] + herizonal[i * BlockSize + x + 32] , store[y * BlockSize + x + 32]);
        store[y * BlockSize + x + 33] = min(vertical[y * BlockSize + i] + herizonal[i * BlockSize + x + 33] , store[y * BlockSize + x + 33]);
    }
    
    write_back(Dist_d , store , x , y , x_her , y_ver , new_n);
}


void input(char* infile) {

    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    new_n = ceil(n , BlockSize) * BlockSize; // for the purpose to make coalsememory
    Dist = (int*) malloc(new_n * new_n *sizeof(int));

    for (int i = 0; i < new_n; ++i) {
        for (int j = 0; j < new_n ; ++j) {
            Dist[i * new_n + j] = (i == j) ? 0 :INF;
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
    /*for(int i = 0 ; i < n ; i++){
        for(int j = 0 ; j < n ; j++){
            printf("%d " , Dist[i * new_n + j]);
        }
        printf("\n");
    }*/
    fclose(outfile);
}



int main(int argc, char* argv[]) {
    input(argv[1]);
    // int BlockSize = 512;
    int* deviceDist;
    printf("%d \n", new_n);
    size_t size_ = sizeof(int) *new_n * new_n;
    cudaHostRegister(Dist,size_, cudaHostRegisterDefault);
    cudaMalloc(&deviceDist, size_);
	cudaMemcpy(deviceDist, Dist, size_, cudaMemcpyHostToDevice);

    int total_block_num = new_n / BlockSize;
    dim3 block_num1(1, 1);
    dim3 block_num2(1, total_block_num);
    dim3 block_num3(total_block_num, total_block_num);
    // dim3 thread_each_block(BlockSize,BlockSize);
    dim3 thread_each_block(16 , BlockSize);
    //dim3 thread_32(32,32);
    // printf("the total block : %d \n" , total_block_num);
    // printf("the n is %d and the new_n is %d \n" ,  n , new_n);
    
    for(int i = 0 ; i < total_block_num ; i++){
        phase1<<<block_num1 , thread_each_block>>>(deviceDist , i , new_n); // phase 1 
        phase2<<<block_num2 , thread_each_block>>>(deviceDist , i , new_n); // phase 2
        phase3<<<block_num3 , thread_each_block>>>(deviceDist , i , new_n); // phase 3
    }
    cudaMemcpy(Dist , deviceDist, size_, cudaMemcpyDeviceToHost); 
    output(argv[2]);
    return 0;
}