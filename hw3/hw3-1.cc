#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
const int INF = ((1 << 30) - 1);
const int V = 6010;

int n, m;
static int Dist[V][V];
int cpu_num;
pthread_barrier_t barrier;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

void* calculate(void* threads){
    int tid = *(int*) threads;
    int gap = n/cpu_num + 1 ; // round up the num
    int from = tid*gap ; 
    int to  = (tid  + 1)*gap; // make sure the 
    if(tid == cpu_num - 1){
        to = n ;
    }
    // printf("from %d to  %d \n" , from , to );
    for(int k = 0 ;  k < n ; k++){
        for(int i = from ; i < to ; i++){
            for(int j = 0 ; j < n ; j++){
                Dist[i][j] = std::min(Dist[i][j], Dist[i][k] + Dist[k][j]);
            }
        }
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    /* set up the pthread */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    cpu_num = CPU_COUNT(&cpu_set);
    pthread_barrier_init(&barrier, NULL, cpu_num);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    input(argv[1]);
    pthread_t threads[cpu_num];
    int id[cpu_num];

    /*for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << Dist[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    for(int i = 0 ; i < cpu_num ; i++){
        // printf("In main: creating thread %d\n", i);
        id[i] = i;
        pthread_create(&threads[i], NULL, calculate, (void*)&id[i] );
    }

    for(int i = 0 ; i < cpu_num ; i++){
        pthread_join(threads[i], NULL);
    }

    /*for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << Dist[i][j] << " ";
        }
        std::cout << std::endl;
    }*/

    output(argv[2]);
    return 0;
}



