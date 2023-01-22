#ifndef JOBTRACKER_H
#define JOBTRACKER_H
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
#include <bits/stdc++.h>
#include <mpi.h>

class JobTracker{
public:
    JobTracker(char **argv , int cpu_num , int mpi_size , std::ofstream* log_out);
    ~JobTracker();
    std::string job_name;
    int num_reducer;
    int delay;
    std::string input_filename;
    int chunk_size;
    std::string locality_config_filename;
    std::string output_dir;
    int num_of_data_trunk;
    int cpu_num;
    int mpi_size;
    std::ofstream* log_out;
    //std::chrono::steady_clock::time_point global_start;
    /*You can assume the data type of the input records is int (i.e., line#), and string (i.e., line text),*/
    void Assign_job();
    void Shuffle();
    void Assign_Reduce();
private:
    std::vector<std::pair<int,int>> map_tasks;
    std::queue<int> reduce_tasks;
};

#endif