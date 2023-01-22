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

__device__ void autogenerate(int* Dist_d , int* target , int x , int y , int new_n){
    target[y * BlockSize + x ] = Dist_d[y * new_n + x];
    target[y * BlockSize + (x + Half) ] = Dist_d[y * new_n + (x + Half)];
    target[(y + Half) * BlockSize + x ] = Dist_d[(y + Half) * new_n + x ];
    target[(y + Half) * BlockSize + (x + Half) ] = Dist_d[(y + Half) * new_n + (x + Half)];
}

__global__ void phase1(int* Dist_d, int round_cnt, int new_n){
    // input shape (32,32)
    __shared__ int store[BlockSize * BlockSize];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_real = threadIdx.x + round_cnt * BlockSize;
    int y_real = threadIdx.y + round_cnt * BlockSize;
    autogenerate(Dist_d ,store , x_real , y_real , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[y * BlockSize + x] = min(store[y * BlockSize + i] + store[i* BlockSize + x] , store[y * BlockSize + x]);
        store[(y + Half) * BlockSize + x] = min(store[(y + Half) * BlockSize + i] + store[i* BlockSize + x] , store[(y + Half) * BlockSize  + x]);
        store[ y  * BlockSize + (x+ Half)] = min(store[y * BlockSize + i  ] + store[ i * BlockSize + (x+ Half)] , store[ y  * BlockSize + (x+ Half)]);
        store[(y + Half)  * BlockSize + (x+ Half)] = min(store[(y + Half) * BlockSize + i  ] + store[ i * BlockSize + (x+ Half)] , store[ (y + Half) * BlockSize + (x+ Half)]);
        __syncthreads();
    }
    Dist_d[y_real * new_n + x_real] = store[y * BlockSize + x];
    Dist_d[ ( y_real + Half )* new_n + x_real] = store[(y + Half) * BlockSize + x];
    Dist_d[y_real * new_n + (x_real + Half) ] = store[y * BlockSize +(x + Half)];
    Dist_d[( y_real + Half )* new_n + x_real + Half ] = store[(y + Half) * BlockSize + x + Half];
}

__global__ void phase2(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize * BlockSize];
    __shared__ int vertical[BlockSize * BlockSize];
    __shared__ int herizonal[BlockSize * BlockSize];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int block_num = blockIdx.y;
    
    int x_ver = threadIdx.x + round_cnt * BlockSize;
    int x_her = threadIdx.x + block_num * BlockSize;
    int y_ver = y + block_num * BlockSize;
    int y_her = y + round_cnt * BlockSize;
    
    // if duplicate with the phase 1
    /*if(block_num == round_cnt ){
        return;
    }*/
    // since we calculate both vertical and herizonal at once ,we can't delete both given one doesn't fulfill
    autogenerate(Dist_d ,store , x_ver , y_her , new_n);
    autogenerate(Dist_d ,herizonal , x_her , y_her , new_n);
    autogenerate(Dist_d ,vertical , x_ver , y_ver , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        vertical[y * BlockSize + x] = min( store[y * BlockSize + i] + vertical[i * BlockSize + x] , vertical[y * BlockSize + x]);
        vertical[(y + Half) * BlockSize + x] = min( store[(y + Half) * BlockSize + i] + vertical[i * BlockSize + x] , vertical[(y + Half) * BlockSize + x]);
        vertical[y * BlockSize + (x + Half)] = min( store[y * BlockSize + i] + vertical[i * BlockSize + (x + Half)] , vertical[y * BlockSize + (x + Half)]);
        vertical[(y + Half) * BlockSize +(x + Half)] = min( store[(y + Half) * BlockSize + i] + vertical[i * BlockSize + (x + Half)] , vertical[(y + Half) * BlockSize + (x + Half)]);

        herizonal[y * BlockSize + x] =  min( herizonal[y * BlockSize + i] + store[i * BlockSize + x] , herizonal[y * BlockSize + x]);
        herizonal[(y + Half) * BlockSize + x] =  min( herizonal[(y + Half) * BlockSize + i] + store[i * BlockSize + x] , herizonal[(y + Half) * BlockSize + x]);
        herizonal[y * BlockSize + (x + Half)] =  min( herizonal[y * BlockSize + i] + store[i * BlockSize + (x + Half)] , herizonal[y * BlockSize + (x + Half)]);
        herizonal[(y + Half) * BlockSize + (x + Half)] =  min( herizonal[(y + Half) * BlockSize + i] + store[i * BlockSize + (x + Half)] , herizonal[(y + Half) * BlockSize + (x + Half)]);
    }

    Dist_d[y_her * new_n + x_ver] = store[y * BlockSize + x] ;
    Dist_d[(y_her + Half) * new_n + x_ver] = store[(y + Half)* BlockSize + x] ;
    Dist_d[y_her * new_n + x_ver + Half] = store[y * BlockSize + ( x + Half)] ;
    Dist_d[(y_her + Half) * new_n + ( x_ver + Half )] = store[(y + Half) * BlockSize + (x + Half) ] ;

    Dist_d[ y_her * new_n + x_ver ] = herizonal[y * BlockSize + x ] ;
    Dist_d[ (y_her + Half) * new_n + x_ver ] = herizonal[(y + Half)* BlockSize + x ];
    Dist_d[ y_her * new_n + (x_ver + Half) ] = herizonal[y * BlockSize + ( x + Half)] ;
    Dist_d[ ( y_her + Half) * new_n + ( x_ver + Half) ] = herizonal[(y + Half) * BlockSize + (x + Half) ];

    Dist_d[ y_ver * new_n + x_ver ] = vertical[y * BlockSize + x];
    Dist_d[ (y_ver+ Half) * new_n + x_ver ] = vertical[(y + Half)* BlockSize + x ];
    Dist_d[ y_ver * new_n + ( x_ver+ Half ) ] = vertical[y * BlockSize + ( x + Half)];
    Dist_d[ ( y_ver + Half) * new_n +  ( x_ver + Half) ] = vertical[(y + Half) * BlockSize + (x + Half) ];
}

__global__ void phase3(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize * BlockSize];
    __shared__ int vertical[BlockSize * BlockSize];
    __shared__ int herizonal[BlockSize * BlockSize];
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;
    /*if(block_x == round_cnt || block_y == round_cnt){
        return;
    }*/
    int x_real = block_x * BlockSize + x;
    int y_real = block_y * BlockSize + y;
    int x_her = x_real;
    int y_her = round_cnt * BlockSize + y;
    int x_ver = round_cnt * BlockSize + x;
    int y_ver = y_real;
    /*if(x_real >= new_n || y_real >= new_n){
        return;
    }*/
    autogenerate(Dist_d , store , x_real , y_real , new_n);
    autogenerate(Dist_d , herizonal , x_her , y_her , new_n);
    autogenerate(Dist_d , vertical , x_ver , y_ver , new_n);

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[y * BlockSize + x] = min(herizonal[y * BlockSize + i] + vertical[i * BlockSize + x] , store[y * BlockSize + x]);
        store[(y + Half) * BlockSize + x] = min(herizonal[(y + Half) * BlockSize + i] + vertical[i * BlockSize + x] , store[(y + Half)* BlockSize + x]);
        store[y * BlockSize + (x + Half)] = min(herizonal[y * BlockSize + i] + vertical[i * BlockSize + (x + Half)] , store[y * BlockSize + (x + Half)]);
        store[(y + Half) * BlockSize + (x + Half)] = min(herizonal[(y + Half) * BlockSize + i] + vertical[i * BlockSize + (x + Half)] , store[(y + Half)* BlockSize + (x + Half)]);
    }
    
    Dist_d[y_real * new_n + x_real] = store[y * BlockSize + x];
    Dist_d[(y_real + Half) * new_n + x_real] = store[(y + Half)* BlockSize + x];
    Dist_d[y_real * new_n + x_real + Half] = store[y * BlockSize + ( x + Half)];
    Dist_d[(y_real + Half) * new_n + ( x_real + Half )] = store[(y + Half) * BlockSize + (x + Half) ];
    
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
    // dim3 thread_each_block(BlockSize, 16 );
    dim3 thread_32(32,32);
    // printf("the total block : %d \n" , total_block_num);
    // printf("the n is %d and the new_n is %d \n" ,  n , new_n);
    
    for(int i = 0 ; i < total_block_num ; i++){
        phase1<<<block_num1 , thread_32>>>(deviceDist , i , new_n); // phase 1 
        phase2<<<block_num2 , thread_32>>>(deviceDist , i , new_n); // phase 2
        phase3<<<block_num3 , thread_32>>>(deviceDist , i , new_n); // phase 3
    }
    cudaMemcpy(Dist , deviceDist, size_, cudaMemcpyDeviceToHost); 
    output(argv[2]);
    return 0;
}