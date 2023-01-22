#include "job_tracker.h"
enum MSG{
    REQUEST , DISPATCH_MAP, REQUEST_REDUCE ,DISPATCH_REDUCE ,
};

JobTracker::JobTracker(char **argv , int cpu_num , int mpi_size , std::ofstream * log_out){
    this->job_name = std::string(argv[1]);
    this->num_reducer = std::stoi(argv[2]);
    this->delay = std::stoi(argv[3]);
    this->input_filename = std::string(argv[4]);
    this->chunk_size = std::stoi(argv[5]);
    this->locality_config_filename = std::string(argv[6]);
    this->output_dir = (argv[7]);
    this->cpu_num = cpu_num;
    this->mpi_size = mpi_size;
    this->num_of_data_trunk = 0;
    this->log_out = log_out;
    // this->global_start = std::chrono::steady_clock::now();
    *this->log_out << std::time(nullptr) << ",Start_Job," << job_name << "," << mpi_size << ","  << cpu_num << ","  << \
    num_reducer << "," << delay << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << std::endl;

    // start read 
    std::ifstream input_file(this->locality_config_filename);
    std::string line;
    while (getline(input_file, line)) {
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos+1)) % (mpi_size-1);
        this->map_tasks.push_back(std::make_pair(chunkID, nodeID));
        num_of_data_trunk++;
    }
    std::cout << "[job tracker] the Job Tracker finish the split \n" ;
}

void JobTracker::Assign_job(){
    int request_rank;
    int send_req[2];
    MPI_Status status; // redundant
    /*for(auto it : map_tasks){
        std::cout <<"[job tracker] : map_task list "<< it.first << " " << it.second << std::endl;
    }*/
    std::chrono::duration<double> time_span;
    std::chrono::steady_clock::time_point start[this->mpi_size];
    for(int i = 0 ; i < this->mpi_size ; i++){
        start[i] = std::chrono::steady_clock::now();
    }
    while(!map_tasks.empty()) {
        MPI_Recv(&request_rank,1,MPI_INT , MPI_ANY_SOURCE , REQUEST , MPI_COMM_WORLD , &status);
        std::pair<int,int> target;
        bool find_the_same = false;
        // iter through the queue
        for(auto it = map_tasks.begin(); it != map_tasks.end(); it++){ // chunkID , nodeID
            if(it->second == request_rank){
                find_the_same = true;
                target = *it;
                map_tasks.erase(it);
                break;
            }
        }
        if(!find_the_same){
            target = map_tasks.front();
            // implement the pop front 
            map_tasks.erase(map_tasks.begin());
        }
        send_req[0] = target.first;
        send_req[1] = target.second;
        std::cout << "[job tracker] send the chunk " << target.first << " with the number " <<  target.second << " to " <<  request_rank << std::endl;
        *this->log_out << std::time(nullptr) << ",Dispatch_MapTask," << request_rank << "," << target.first  << "\n";
        MPI_Send(&send_req, 2 , MPI_INT , request_rank , MSG::DISPATCH_MAP,MPI_COMM_WORLD);
    }
    // end up sending
    std::cout << "[job tracker] all map_job done \n" ;
    send_req[0] = -1;
    send_req[1] = -1;
    for(int i = 1; i < mpi_size ; i++){
        MPI_Recv( &request_rank, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST, MPI_COMM_WORLD, &status);
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start[request_rank]);
        *this->log_out << std::time(nullptr) << ",Complete_MapTask," << request_rank << "," << time_span.count() <<  "\n";
        MPI_Send( &send_req, 2 , MPI_INT , request_rank , MSG::DISPATCH_MAP,MPI_COMM_WORLD);
    }

}

void JobTracker::Shuffle(){
    std::cout << "[job tracker] : Start shuffle "<< std::endl;
    int kv_count = 0;
    int test_cnt = 0;
    std::vector<std::vector<std::pair<std::string,int>>> data( num_reducer , std::vector<std::pair<std::string,int>>());
    std::string word;
    int count;
    int hash;

    std::chrono::duration<double> time_span;
    auto start = std::chrono::steady_clock::now();

    for(int i = 1 ; i <= this->num_of_data_trunk ; i++){
        std::ifstream in("./mapper_intermediate_" + std::to_string(i)  + ".txt");
        while (in >> word >> count >> hash) {
            kv_count++;
            data[hash].push_back(std::make_pair(word, count));
        }
        in.close();
    }
    *this->log_out << std::time(nullptr) << ",Start_Shuffle," <<  kv_count <<  "\n";

    std::cout << "[job tracker] : KV Total count "<< kv_count <<std::endl;
    for(int i = 0 ; i < this->num_reducer ; i++){
        std::ofstream out("./mapper_reducer_" + std::to_string(i)  + ".txt");
        for(auto it : data[i]){
            test_cnt++;
            out << it.first << " " << it.second << "\n";
        }
        out.close();
        reduce_tasks.push(i);
    }
    std::cout << "[job tracker] : Test Total count "<< test_cnt <<std::endl;
    auto end = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    *this->log_out << std::time(nullptr) << ",Finish_Shuffle," <<  (int)time_span.count() <<  "\n";
}

void JobTracker::Assign_Reduce(){
    int request_rank;
    int target;
    MPI_Status status;
    std::chrono::duration<double> time_span;
    std::chrono::steady_clock::time_point start[this->mpi_size];
    for(int i = 0 ; i < this->mpi_size ; i++){
        start[i] = std::chrono::steady_clock::now();
    }
    while(!reduce_tasks.empty()) {
        MPI_Recv(&request_rank,1,MPI_INT , MPI_ANY_SOURCE , MSG::REQUEST_REDUCE , MPI_COMM_WORLD , &status);
        target = reduce_tasks.front();
        reduce_tasks.pop();
        std::cout << "[job tracker] send the reduce " << target << " to " <<  request_rank << std::endl;
        *this->log_out << std::time(nullptr) << ",Dispatch_ReduceTask," << request_rank << "," << target  << "\n";
        MPI_Send(&target, 1 , MPI_INT , request_rank , MSG::DISPATCH_REDUCE,MPI_COMM_WORLD);
    }
    // end up sending
    std::cout << "[job tracker] Reduce : all reduce done \n" ;
    target = -1;
    for(int i = 1; i < mpi_size ; i++){
        MPI_Recv( &request_rank, 1, MPI_INT, MPI_ANY_SOURCE, MSG::REQUEST_REDUCE, MPI_COMM_WORLD, &status);
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - start[request_rank]);
        *this->log_out << std::time(nullptr) << ",Complete_ReduceTask," << request_rank << "," << (int)time_span.count()/1000000 <<  "\n";
        MPI_Send( &target, 1 , MPI_INT , request_rank , MSG::DISPATCH_REDUCE,MPI_COMM_WORLD);
    }
}

JobTracker::~JobTracker(){
    
    std::cout << "[job tracker] called end" << std::endl;
}
