#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <omp.h> 
int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0 , x;
	int cnt = 0;
	unsigned long long omp_threads, omp_thread;
	#pragma omp parallel 
	{
		omp_threads = omp_get_num_threads();
	}
	#pragma omp parallel num_threads(omp_threads) private(x)  reduction(+:pixels,cnt)
    {
		#pragma omp for schedule(dynamic , 10000) nowait
		for (x = 0; x < r; x++){
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			pixels += y;
		}
		// printf("mod ! \n");
		pixels %= k;
    }
	// printf("reached %d \n" , cnt);
	// printf("going yo output the answer \n");
	printf("%llu\n", (4 * pixels) % k);
}
