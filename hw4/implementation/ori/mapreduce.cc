#include <iostream>
#include <deque>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <queue>
#include <pthread.h>
#include <chrono>
#include <ctime>
#include "job_tracker.h"
#include "task_tracker.h"
#include <mpi.h>
// include self define class


int main(int argc, char **argv){
    if (argc != 8) {
		fprintf(stderr, "must provide exactly 7 arguments!\n");
		return 1;
	}
    std::string job_name = std::string(argv[1]);
    int num_reducer = std::stoi(argv[2]);
    int delay = std::stoi(argv[3]);
    std::string input_filename = std::string(argv[4]);
    int chunk_size = std::stoi(argv[5]);
    std::string locality_config_filename = std::string(argv[6]);
    std::string output_dir = std::string(argv[7]);
    std::ofstream log_file(output_dir + "/" + job_name + "-log.out");
    std::chrono::duration<double> time_span;
    auto start = std::chrono::steady_clock::now();

    // set up cpu and mpi 
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int cpu_num = CPU_COUNT(&cpu_set);
    int rank , size ; 
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("initialize with rank %d and total size %d \n",rank,size);
    std::ofstream* log_out = new std::ofstream(output_dir + "/" + job_name + "-log.out");
    
    if(rank == size - 1){
        // show the default config 
        std::cout << "[job tracker] job name : " << job_name << " \nnum_reducer : " << num_reducer << \
        "\ndelay :" << delay << "\ninput_filename : " << input_filename << "\nchunk_size : " << chunk_size << \
        "\nlocality_config : " << locality_config_filename << "\noutput_dir : " << output_dir  << std::endl;
        // assign the JobTracker for the first rank
        JobTracker job_Tracker(argv , cpu_num , size , log_out);
        job_Tracker.Assign_job();
        job_Tracker.Shuffle();
        job_Tracker.Assign_Reduce();
    }
    else{
        TaskTracker task_Tracker(argv , cpu_num , size , rank );
        task_Tracker.required_job();
        task_Tracker.required_reduce();
    }
    MPI_Finalize();
    auto end = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    *log_out << std::time(nullptr) << ",Finish_Job," << time_span.count() <<  "\n";
    log_out->close();
    return 0;
}