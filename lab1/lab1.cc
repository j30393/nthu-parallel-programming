#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	int rank, size;
    unsigned long long final_ans = 0;
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	unsigned long long pixels = 0;
	for(unsigned long long  i = rank ; i < r ; i+= size ){
        //printf("I am in %d and running %i task\n" , rank , i);
        unsigned long long temp = ceil(sqrtl(r*r - i*i));
        temp %= k;
        pixels += temp;
        pixels %= k;
    }
	MPI_Reduce(&pixels , &final_ans , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , 0 , MPI_COMM_WORLD);
	if(rank == 0){
		final_ans %= k;
		printf("%llu\n", (4 * final_ans) % k);
	}
	MPI_Finalize();
	return 0;
}
