#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char *argv[])
{
    int rank, size;
    long final_ans = 0;
    long r = strtol(argv[1], NULL, 10);
    int k = strtol(argv[2], NULL, 10);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    long ans = 0;
    
    MPI_Reduce(&ans , &final_ans , 1 , MPI_LONG , MPI_SUM , 0 , MPI_COMM_WORLD);
    if(rank == 0){
        final_ans *= 4;
        final_ans %= k;
        printf("%ld\n", final_ans);
    }
    MPI_Finalize();
    return 0;
}
