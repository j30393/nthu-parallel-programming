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

__global__ void phase1(int* Dist_d, int round_cnt, int new_n){
    // input shape (32,32)
    __shared__ int store[BlockSize * BlockSize];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int x_real = threadIdx.x + round_cnt * BlockSize;
    int y_real = threadIdx.y + round_cnt * BlockSize;
    
    store[x * BlockSize + y] = Dist_d[x_real * new_n + y_real ];
    store[(x + Half) * BlockSize + y] = Dist_d[ ( x_real + Half )* new_n + y_real];
    store[x * BlockSize + (y + Half )] = Dist_d[x_real * new_n + ( y_real + Half ) ] ;
    store[(x + Half) * BlockSize + y + Half] = Dist_d[( x_real + Half )* new_n + y_real + Half ] ;
    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[x * BlockSize + y] = min(store[x * BlockSize + i] + store[i* BlockSize + y] , store[x * BlockSize + y]);
        store[(x + Half) * BlockSize + y] = min(store[(x + Half) * BlockSize + i] + store[i* BlockSize + y] , store[(x + Half) * BlockSize  + y]);
        store[ x  * BlockSize + (y+ Half)] = min(store[x * BlockSize + i  ] + store[ i * BlockSize + (y+ Half)] , store[ x  * BlockSize + (y+ Half)]);
        store[(x + Half)  * BlockSize + (y+ Half)] = min(store[(x + Half) * BlockSize + i  ] + store[ i * BlockSize + (y+ Half)] , store[ (x + Half) * BlockSize + (y+ Half)]);
        __syncthreads();
    }
    Dist_d[x_real * new_n + y_real] = store[x * BlockSize + y];
    Dist_d[ ( x_real + Half )* new_n + y_real] = store[(x + Half) * BlockSize + y];
    Dist_d[x_real * new_n + (y_real + Half) ] = store[x * BlockSize +(y + Half)];
    Dist_d[( x_real + Half )* new_n + y_real + Half ] = store[(x + Half) * BlockSize + y + Half];
}

__global__ void phase2(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize][BlockSize];
    __shared__ int vertical[BlockSize][BlockSize];
    __shared__ int herizonal[BlockSize][BlockSize];
    int x = threadIdx.x;
    int block_num = blockIdx.y;
    int y_off = threadIdx.y * 4;
    int x_ver = threadIdx.x + round_cnt * BlockSize;
    int x_her = threadIdx.x + block_num * BlockSize;
    int y_ver = y_off + block_num * BlockSize;
    int y_her = y_off + round_cnt * BlockSize;
    
    // if duplicate with the phase 1
    /*if(block_num == round_cnt ){
        return;
    }*/
    // since we calculate both vertical and herizonal at once ,we can't delete both given one doesn't fulfill
    #pragma unroll 4
    for(int i = 0 ; i < 4 ; i++){
        store[x][y_off + i] = Dist_d[x_ver * new_n + y_her + i];
        herizonal[x][y_off + i] = Dist_d[ x_her * new_n + y_her + i];
        vertical[x][y_off + i] = Dist_d[ x_ver * new_n + y_ver + i];
    }
    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        #pragma unroll 4
        for(int j = 0 ; j < 4 ; j++){
            vertical[x][y_off + j] = min( store[x][i] + vertical[i][y_off + j] , vertical[x][y_off + j]);
            herizonal[x][y_off + j] =  min( herizonal[x][i] + store[i][y_off + j] , herizonal[x][y_off + j]);
        }
    }
    #pragma unroll 4
    for(int i = 0 ; i < 4 ; i++){
        Dist_d[x_her * new_n + y_her + i] = herizonal[x][y_off + i];
        Dist_d[x_ver * new_n + y_ver + i] = vertical[x][y_off + i];
    }
    
}

/*__global__ void phase2(int* Dist_d, int round_cnt, int new_n){
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
    if(block_num == round_cnt ){
        return;
    }
    // since we calculate both vertical and herizonal at once ,we can't delete both given one doesn't fulfill

    store[x * BlockSize + y] = Dist_d[x_ver * new_n + y_her];
    store[(x + Half)* BlockSize + y] = Dist_d[(x_ver + Half) * new_n + y_her];
    store[x * BlockSize + ( y + Half)] = Dist_d[x_ver * new_n + y_her + Half];
    store[(x + Half) * BlockSize + (y + Half) ] = Dist_d[(x_ver + Half) * new_n + ( y_her + Half )];

    herizonal[x * BlockSize + y ] = Dist_d[ x_her * new_n + y_her ];
    herizonal[(x + Half)* BlockSize + y ] = Dist_d[ (x_her + Half) * new_n + y_her ];
    herizonal[x * BlockSize + ( y + Half)] = Dist_d[ x_her * new_n + (y_her + Half) ];
    herizonal[(x + Half) * BlockSize + (y + Half) ] = Dist_d[ ( x_her + Half) * new_n + ( y_her + Half) ];

    vertical[x * BlockSize + y] = Dist_d[ x_ver * new_n + y_ver ];
    vertical[(x + Half)* BlockSize + y ] = Dist_d[ (x_ver+ Half) * new_n + y_ver ];
    vertical[x * BlockSize + ( y + Half)] = Dist_d[ x_ver * new_n + ( y_ver+ Half ) ];
    vertical[(x + Half) * BlockSize + (y + Half) ] = Dist_d[ ( x_ver + Half) * new_n +  ( y_ver + Half) ];

    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        vertical[x * BlockSize + y] = min( store[x * BlockSize + i] + vertical[i * BlockSize + y] , vertical[x * BlockSize + y]);
        vertical[(x + Half) * BlockSize + y] = min( store[(x + Half) * BlockSize + i] + vertical[i * BlockSize + y] , vertical[(x + Half) * BlockSize + y]);
        vertical[x * BlockSize + (y + Half)] = min( store[x * BlockSize + i] + vertical[i * BlockSize + (y + Half)] , vertical[x * BlockSize + (y + Half)]);
        vertical[(x + Half) * BlockSize +(y + Half)] = min( store[(x + Half) * BlockSize + i] + vertical[i * BlockSize + (y + Half)] , vertical[(x + Half) * BlockSize + (y + Half)]);

        herizonal[x * BlockSize + y] =  min( herizonal[x * BlockSize + i] + store[i * BlockSize + y] , herizonal[x * BlockSize + y]);
        herizonal[(x + Half) * BlockSize + y] =  min( herizonal[(x + Half) * BlockSize + i] + store[i * BlockSize + y] , herizonal[(x + Half) * BlockSize + y]);
        herizonal[x * BlockSize + (y + Half)] =  min( herizonal[x * BlockSize + i] + store[i * BlockSize + (y + Half)] , herizonal[x * BlockSize + (y + Half)]);
        herizonal[(x + Half) * BlockSize + (y + Half)] =  min( herizonal[(x + Half) * BlockSize + i] + store[i * BlockSize + (y + Half)] , herizonal[(x + Half) * BlockSize + (y + Half)]);
    }

    Dist_d[x_ver * new_n + y_her] = store[x * BlockSize + y] ;
    Dist_d[(x_ver + Half) * new_n + y_her] = store[(x + Half)* BlockSize + y] ;
    Dist_d[x_ver * new_n + y_her + Half] = store[x * BlockSize + ( y + Half)] ;
    Dist_d[(x_ver + Half) * new_n + ( y_her + Half )] = store[(x + Half) * BlockSize + (y + Half) ] ;

    Dist_d[ x_her * new_n + y_her ] = herizonal[x * BlockSize + y ] ;
    Dist_d[ (x_her + Half) * new_n + y_her ] = herizonal[(x + Half)* BlockSize + y ];
    Dist_d[ x_her * new_n + (y_her + Half) ] = herizonal[x * BlockSize + ( y + Half)] ;
    Dist_d[ ( x_her + Half) * new_n + ( y_her + Half) ] = herizonal[(x + Half) * BlockSize + (y + Half) ];

    Dist_d[ x_ver * new_n + y_ver ] = vertical[x * BlockSize + y];
    Dist_d[ (x_ver+ Half) * new_n + y_ver ] = vertical[(x + Half)* BlockSize + y ];
    Dist_d[ x_ver * new_n + ( y_ver+ Half ) ] = vertical[x * BlockSize + ( y + Half)];
    Dist_d[ ( x_ver + Half) * new_n +  ( y_ver + Half) ] = vertical[(x + Half) * BlockSize + (y + Half) ];
}*/


/*__global__ void phase3(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize * BlockSize];
    __shared__ int vertical[BlockSize * BlockSize];
    __shared__ int herizonal[BlockSize * BlockSize];
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
    store[x * BlockSize + y] = Dist_d[x_real * new_n + y_real];
    store[(x + Half)* BlockSize + y] = Dist_d[(x_real + Half) * new_n + y_real];
    store[x * BlockSize + ( y + Half)] = Dist_d[x_real * new_n + y_real + Half];
    store[(x + Half) * BlockSize + (y + Half) ] = Dist_d[(x_real + Half) * new_n + ( y_real + Half )];

    herizonal[x * BlockSize + y ] = Dist_d[ x_her * new_n + y_her ];
    herizonal[(x + Half)* BlockSize + y ] = Dist_d[ (x_her + Half) * new_n + y_her ];
    herizonal[x * BlockSize + ( y + Half)] = Dist_d[ x_her * new_n + (y_her + Half) ];
    herizonal[(x + Half) * BlockSize + (y + Half) ] = Dist_d[ ( x_her + Half) * new_n + ( y_her + Half) ];

    vertical[x * BlockSize + y] = Dist_d[ x_ver * new_n + y_ver ];
    vertical[(x + Half)* BlockSize + y ] = Dist_d[ (x_ver+ Half) * new_n + y_ver ];
    vertical[x * BlockSize + ( y + Half)] = Dist_d[ x_ver * new_n + ( y_ver+ Half ) ];
    vertical[(x + Half) * BlockSize + (y + Half) ] = Dist_d[ ( x_ver + Half) * new_n +  ( y_ver + Half) ];
    
    __syncthreads();
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        store[x * BlockSize + y] = min(herizonal[x * BlockSize + i] + vertical[i * BlockSize + y] , store[x * BlockSize + y]);
        store[(x + Half) * BlockSize + y] = min(herizonal[(x + Half) * BlockSize + i] + vertical[i * BlockSize + y] , store[(x + Half)* BlockSize + y]);
        store[x * BlockSize + (y + Half)] = min(herizonal[x * BlockSize + i] + vertical[i * BlockSize + (y + Half)] , store[x * BlockSize + (y + Half)]);
        store[(x + Half) * BlockSize + (y + Half)] = min(herizonal[(x + Half) * BlockSize + i] + vertical[i * BlockSize + (y + Half)] , store[(x + Half)* BlockSize + (y + Half)]);
    }
    
    Dist_d[x_real * new_n + y_real] = store[x * BlockSize + y];
    Dist_d[(x_real + Half) * new_n + y_real] = store[(x + Half)* BlockSize + y];
    Dist_d[x_real * new_n + y_real + Half] = store[x * BlockSize + ( y + Half)];
    Dist_d[(x_real + Half) * new_n + ( y_real + Half )] = store[(x + Half) * BlockSize + (y + Half) ];
    
}*/

__global__ void phase3(int* Dist_d, int round_cnt, int new_n){
    __shared__ int store[BlockSize][BlockSize];
    __shared__ int vertical[BlockSize][BlockSize];
    __shared__ int herizonal[BlockSize][BlockSize];
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int x = threadIdx.x;
    int y_off = threadIdx.y * 4;
    /*if(block_x == round_cnt || block_y == round_cnt){
        return;
    }*/
    int x_real = block_x * BlockSize + x;
    int y_real = block_y * BlockSize + y_off;
    int x_her = x_real;
    int y_her = round_cnt * BlockSize + y_off;
    int x_ver = round_cnt * BlockSize + x;
    int y_ver = y_real;
    int store_place = x_real * new_n + y_real;
    
    /*if(x_real >= new_n || y_real >= new_n){
        return;
    }*/
    #pragma unroll 4
    for(int i = 0 ; i < 4 ; i++){
        store[y_off + i][x] = Dist_d[store_place + i];
        vertical[y_off + i][x] = Dist_d[x_ver * new_n + y_ver + i];
        herizonal[y_off + i][x] = Dist_d[x_her * new_n + y_her + i];
    }
    __syncthreads();
    int ans[4] = {store[y_off + 0][x] , store[y_off + 1][x] , store[y_off + 2][x] , store[y_off + 3][x]};
    #pragma unroll 32
    for(int i = 0 ; i < BlockSize ; i++){
        #pragma unroll 4
        for(int j = 0 ; j < 4 ; j++){
            ans[j] = min(herizonal[i][x] + vertical[y_off + j][i] , ans[j]);
        }
    }
    #pragma unroll 4
    for(int i = 0 ; i < 4 ; i++){
        Dist_d[store_place + i] = ans[i];
    }
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
    dim3 thread_each_block(BlockSize, 16 );
    dim3 thread_32(32,32);
    // printf("the total block : %d \n" , total_block_num);
    // printf("the n is %d and the new_n is %d \n" ,  n , new_n);
    
    for(int i = 0 ; i < total_block_num ; i++){
        phase1<<<block_num1 , thread_32>>>(deviceDist , i , new_n); // phase 1 
        phase2<<<block_num2 , thread_each_block>>>(deviceDist , i , new_n); // phase 2
        phase3<<<block_num3 , thread_each_block>>>(deviceDist , i , new_n); // phase 3
    }
    cudaMemcpy(Dist , deviceDist, size_, cudaMemcpyDeviceToHost); 
    output(argv[2]);
    return 0;
}