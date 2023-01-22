#ifndef TASKTRACKER_H
#define TASKTRACKER_H
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
#include <sstream>
#include <queue>
#include <pthread.h>
#include <chrono>
#include <ctime>
#include <functional>
#include "job_tracker.h"
#include <mpi.h>
// #include "thread_pool.h"

// the above code is given by TA
/*struct Map_Args {
    Map_Args(int chunk_num){
        this->chunk_num = chunk_num;
    }
    int chunk_num;
};*/

class TaskTracker{
public:
    TaskTracker(char** argv , int cpu_num , int size , int rank);
    ~TaskTracker();
    std::string job_name;
    int num_reducer;
    int delay;
    std::string input_filename;
    int chunk_size;
    std::string locality_config_filename;
    std::string output_dir;
    int cpu_num;
    int mpi_size;
    int which_node;
    int job_tracker_node;
    void required_job();
    static void* map_pool(void* input) ;
    static void* reduce_pool(void* input) ;
    void required_reduce();
    /*static void* map_function(void* arg){
        
    };*/
    // sync tool 
private:
    pthread_t* Mapper_threads;
    pthread_t* Reducer_threads;
    pthread_mutex_t Mapper_mutex_com;
    pthread_cond_t Mapper_cond_com;
    pthread_mutex_t Mapper_mutex_job;
    pthread_cond_t Mapper_cond_job;
    pthread_mutex_t Reducer_mutex_com;
    pthread_cond_t Reducer_cond_com;
    pthread_mutex_t Reducer_mutex_job;
    pthread_cond_t Reducer_cond_job;
    std::queue<int> map_task_queue;
    std::queue<int> assigned_task;
    int Map_thread_cnt;
    bool Reduce_thread_busy;
};


#endif