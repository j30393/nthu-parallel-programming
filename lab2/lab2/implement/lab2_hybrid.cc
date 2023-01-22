#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long ans = 0;
	unsigned long long pixels = 0 , x;
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks, omp_threads, omp_thread;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	#pragma omp parallel 
	{
		omp_threads = omp_get_num_threads();
		// omp_thread = omp_get_thread_num();
		// printf("Hello %s: rank %2d/%2d, thread %2d/%2d\n", hostname, mpi_rank, mpi_ranks, omp_thread, omp_threads);
		
	}
	#pragma omp parallel num_threads(omp_threads) private(x)  reduction(+:pixels)
    {
		#pragma omp for schedule(dynamic , 500000) nowait
		for (x = mpi_rank; x < r; x += mpi_ranks){
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			pixels += y;
			// printf("x is : %d and rank is %d \n" , x , mpi_rank);
		}
		// printf("mod ! \n");
		pixels %= k;
    }

	MPI_Reduce(&pixels , &ans , 1 , MPI_UNSIGNED_LONG_LONG , MPI_SUM , 0 , MPI_COMM_WORLD);
	if(mpi_rank == 0){
		printf("%llu\n", (4 * ans) % k);
	}
	MPI_Finalize();
}
