#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
unsigned long long r, k  , ncpus;

struct pair_ {
    long long first;
    unsigned long long second;
};

void* calculate(void* threadid){
	pair_* tid = static_cast<pair_*>(threadid);
	unsigned long long x_first = tid->first;
	unsigned long long x_second = 0;
	// unsigned long long int y_sq = r*r;
	for (unsigned long long x = x_first ; x < r; x += ncpus) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		x_second += y;
		// x_second %= k; -> adding this line cause tle 
	}
	// printf("we have %llu in the pixel in thread %llu \n" , x_second , x_first );
	tid->second = x_second;
	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);
	pthread_t threads[ncpus];
	pair_ ID[ncpus];
	unsigned long long pixels = 0;
    int t;
	for(int i = 0 ; i < ncpus ; i++){
		ID[i].first = i;
		ID[i].second = 0;
		pthread_create(&threads[i], NULL, calculate, (void*)&ID[i]);
		/*if(rc){
			printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }*/
	}
	for(int i = 0 ; i < ncpus ; i++){
		pthread_join(threads[i], NULL);
		pixels += ID[i].second;
		pixels %= k;
	}
	// printf("I'm here");
	printf("%llu\n", (4 * pixels) % k);
}
