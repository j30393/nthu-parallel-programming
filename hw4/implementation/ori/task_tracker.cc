#include "task_tracker.h"
enum MSG{
    REQUEST , DISPATCH_MAP, REQUEST_REDUCE , DISPATCH_REDUCE ,
};

TaskTracker::TaskTracker(char** argv , int cpu_num , int size , int rank){
    this->job_name = std::string(argv[1]);
    this->num_reducer = std::stoi(argv[2]);
    this->delay = std::stoi(argv[3]);
    this->input_filename = std::string(argv[4]);
    this->chunk_size = std::stoi(argv[5]);
    this->locality_config_filename = std::string(argv[6]);
    this->output_dir = (argv[7]);
    this->cpu_num = cpu_num;
    this->mpi_size = size;
    this->which_node = rank;
    this->job_tracker_node = size - 1;
    this->Map_thread_cnt = 0;
    this->Reduce_thread_busy = false;
    /*Each tasktracker will create CPUS-1 mapper threads*/ 
    this->Mapper_threads = new pthread_t[cpu_num- 1];
    this->Reducer_threads = new pthread_t;
    pthread_mutex_init(&Mapper_mutex_com, nullptr);
    pthread_cond_init(&Mapper_cond_com, nullptr);
    pthread_mutex_init(&Mapper_mutex_job, nullptr);
    pthread_cond_init(&Mapper_cond_job, nullptr);
    pthread_mutex_init(&Reducer_mutex_com, nullptr);
    pthread_cond_init(&Reducer_cond_com, nullptr);
    pthread_mutex_init(&Reducer_mutex_job, nullptr);
    pthread_cond_init(&Reducer_cond_job, nullptr);
    std::cout << "[Task tracker] " << "initialize the pool " << std::endl;
    /*this->Mapper_Pool->start();
    this->Reducer_Pool->start();*/
}

std::map<int, std::string> Input_Split(int chunk , int read_lines , std::string filename){
    std::map<int, std::string> buffer;
    std::ifstream input(filename);
    // ignore 
    std::string line;
    for(int i = 0 ; i < read_lines*(chunk - 1) ; i++){
        std::getline(input, line);
    }
    for(int i = 0 ; i < read_lines ; i++){
        std::getline(input, line);
        buffer.insert({chunk * read_lines + i , line});
    }
    return buffer;
}

std::map<std::string, int> MapFunction(std::map<int, std::string> input) {
    std::map<std::string, int> output;
    int cnt = 0;
    for (const auto& record : input) {
        std::string line = record.second;
        std::istringstream iss(line);
        std::string word;
        while (iss >> word) {
            cnt++;
            if (output.count(word) > 0) {
                output[word]++;
            } else {
                output[word] = 1;
            }
        }
    }
    //std::cout <<  "[Function Info] : MapFunction : " <<  cnt << std::endl;
    return output;
}

int hash_function(std::string input , int num_reducer){
    std::hash<std::string> hasher;
    return hasher(input) % num_reducer;
}

void* TaskTracker::map_pool(void *input){
    // keep on working working
    TaskTracker* Self = (TaskTracker*)input;
    std::chrono::steady_clock::time_point start, end;
    int to_work_on = -1;
    while(true){
        pthread_mutex_lock(&Self->Mapper_mutex_job);
            while(Self->map_task_queue.empty()){
                pthread_cond_wait(&Self->Mapper_cond_job, &Self->Mapper_mutex_job);
            }
            to_work_on = Self->map_task_queue.front();
            Self->map_task_queue.pop();
        pthread_mutex_unlock(&Self->Mapper_mutex_job);
        if(to_work_on != -1){
            std::cout << "[Task tracker] Node : " << Self->which_node <<  " Work on :" << to_work_on << std::endl;
            auto return_from_split = Input_Split(to_work_on , Self->chunk_size , Self->input_filename);
            auto from_Map = MapFunction(return_from_split);

            std::ofstream out("./mapper_intermediate_" + std::to_string(to_work_on)  + ".txt");
            // output format word , value , hash value
            for(auto it : from_Map){
                out <<  it.first << " " << it.second << " "<< hash_function(it.first, Self->num_reducer) << "\n";
            }
            out.close();
            pthread_mutex_lock(&Self->Mapper_mutex_job);
            Self->Map_thread_cnt--;
            usleep(2000);
            pthread_cond_signal(&Self->Mapper_cond_com);
            pthread_mutex_unlock(&Self->Mapper_mutex_job);
        }
        else{
            std::cout << "[Task tracker] " << Self->which_node << " Received the end notation "  << std::endl;
            break;
        }
    }
    pthread_exit(NULL);
}

void TaskTracker::required_job(){
    for(int i = 0 ; i < cpu_num - 1; i++){
        pthread_create(&this->Mapper_threads[i], nullptr, TaskTracker::map_pool, (void*)this);
    }
    int recv_arr[2];
    while(1){
        pthread_mutex_lock(&Mapper_mutex_com);
        if(Map_thread_cnt >= cpu_num - 1 ){
            pthread_cond_wait(&Mapper_cond_com, &Mapper_mutex_com);
        }
        pthread_mutex_unlock(&Mapper_mutex_com);
        MPI_Send(&(this->which_node),1,MPI_INT , job_tracker_node , MSG::REQUEST , MPI_COMM_WORLD);
        MPI_Recv(&recv_arr, 2 , MPI_INT , job_tracker_node , MSG::DISPATCH_MAP , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
        if(recv_arr[0] == -1){
            pthread_mutex_lock(&Mapper_mutex_com);
            std::cout << "[Task tracker] " << this->which_node << " : finish receiving "  << std::endl;
            // to indicate the tasks are over
            this->map_task_queue.push(recv_arr[0]);
            usleep(2000);
            pthread_cond_signal(&Mapper_cond_job);
            pthread_mutex_unlock(&Mapper_mutex_com);
            break;
        }
        else{
            // Map_Args* arg = new Map_Args(recv_arr[0]);
            if(recv_arr[1] != this->which_node){
                sleep(delay);
                std::cout << "[Task tracker] delay !!!!" << this->which_node << " : received " << recv_arr[0] << std::endl;
            }
            else{
                std::cout << "[Task tracker] " << this->which_node << " : received " << recv_arr[0] << std::endl;
            }
            pthread_mutex_lock(&Mapper_mutex_com);
            // ( cpu_num - 1 ) mapper threads
            this->map_task_queue.push(recv_arr[0]);
            Map_thread_cnt++;
            // after the put into the queue wake up the processor
            usleep(2000);
            pthread_cond_signal(&Mapper_cond_job);
            std::cout << "[Task tracker] " << this->which_node << " : signal the task " << recv_arr[0] << std::endl;
            pthread_mutex_unlock(&Mapper_mutex_com);
            // Mapper_Pool->addTask(new ThreadPoolTask(&map_function, (void*)arg));
        }
    }
}


std::vector<std::pair<std::string,int>> extract_data(int reducer , std::string output_dir){
    std::vector<std::pair<std::string,int>> output;
    std::string word;
    int count;
    int word_cnt;
    std::ifstream in("./mapper_reducer_" + std::to_string(reducer)  + ".txt");
    while (in >> word >> count ) {
        word_cnt++;
        output.push_back(std::make_pair(word,count));
    }
    in.close();
    std::cout << "[extract_data] :size " << word_cnt << "\n";
    return output;
}

std::vector<std::pair<std::string, int>> Sorting_function(std::vector<std::pair<std::string, int>> input) {
    std::sort(input.begin(), input.end(), [](const std::pair<std::string,int> &a, const std::pair<std::string,int> &b) {
        return a.first < b.first;
    });
    return input;
}

//  output format map<string, Item_list>

std::map<std::string, std::vector<std::pair<std::string, int>>> group_function(std::vector<std::pair<std::string, int>> input){
    std::map<std::string, std::vector<std::pair<std::string, int>>> output;
    std::cout << "[group_function] :size " <<input.size() << "\n";
    for(auto element: input) {
        if(output.find(element.first) == output.end())
            output[element.first] = std::vector<std::pair<std::string, int>>();
        output[element.first].push_back(element);
    }
    return output;
}

std::vector<std::pair<std::string, int>> reduce_function(const std::map<std::string, std::vector<std::pair<std::string, int>>> &input) {
    std::vector<std::pair<std::string, int>> output;
    for (auto it = input.begin(); it != input.end(); it++) {
        std::string key = it->first;
        int count = 0;
        auto value = it->second;
        for (const auto &p: value)
            count += p.second;
        output.push_back({key, count});
    }
    return output;
}

void* TaskTracker::reduce_pool(void *input){
    // keep on working 
    TaskTracker* Self = (TaskTracker*)input;
    std::chrono::steady_clock::time_point start, end;
    int to_work_on = -1;
    while(true){
        pthread_mutex_lock(&Self->Reducer_mutex_job);
            while(Self->assigned_task.empty()){
                pthread_cond_wait(&Self->Reducer_cond_job, &Self->Reducer_mutex_job);
            }
            to_work_on = Self->assigned_task.front();
            Self->assigned_task.pop();
        pthread_mutex_unlock(&Self->Reducer_mutex_job);
        if(to_work_on != -1){
            std::cout << "[Task tracker] Node : " << Self->which_node <<  " Work on Reduce:" << to_work_on << std::endl;
            auto data = extract_data(to_work_on , Self->output_dir);
            data = Sorting_function(data);
            auto group_results = group_function(data);
            auto final_output = reduce_function(group_results);
            std::ofstream out(Self->output_dir + "/" + Self->job_name + "-" + std::to_string(to_work_on) + ".out");
            for(auto it : final_output){
                out << it.first << " " << it.second << "\n";
            }
            out.close();
            pthread_mutex_lock(&Self->Reducer_mutex_job);
            Self->Reduce_thread_busy = false;
            usleep(2000);
            pthread_cond_signal(&Self->Reducer_cond_com);
            pthread_mutex_unlock(&Self->Reducer_mutex_job);
        }
        else{
            std::cout << "[Task tracker] Reducer : " << Self->which_node << " Received the end notation "  << std::endl;
            break;
        }
    }
    pthread_exit(NULL);
}


void TaskTracker::required_reduce(){
    pthread_create(&this->Reducer_threads[0], nullptr, TaskTracker::reduce_pool, (void*)this);
    int recv_task;
    while(1){
        pthread_mutex_lock(&Reducer_mutex_com);
        if( Reduce_thread_busy == true ){
            pthread_cond_wait(&Reducer_cond_com, &Reducer_mutex_com);
        }
        pthread_mutex_unlock(&Reducer_mutex_com);
        MPI_Send(&(this->which_node),1,MPI_INT , job_tracker_node , MSG::REQUEST_REDUCE , MPI_COMM_WORLD);
        MPI_Recv(&recv_task, 1 , MPI_INT , job_tracker_node , MSG::DISPATCH_REDUCE , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
        if(recv_task == -1){
            pthread_mutex_lock(&Reducer_mutex_com);
            std::cout << "[Task tracker] " << this->which_node << " : finish receiving Reduce "  << std::endl;
            // to indicate the tasks are over
            this->assigned_task.push(recv_task);
            Reduce_thread_busy = true;
            pthread_cond_signal(&Reducer_cond_job);
            pthread_mutex_unlock(&Reducer_mutex_com);
            usleep(2000);
            break;
        }
        else{
            pthread_mutex_lock(&Reducer_mutex_com);
            std::cout << "[Task tracker] " << this->which_node << " : received Reduce " << recv_task << std::endl;
            this->assigned_task.push(recv_task);
            Reduce_thread_busy = true;
            usleep(2000);
            pthread_cond_signal(&Reducer_cond_job);
            pthread_mutex_unlock(&Reducer_mutex_com);
        }
    }
}

TaskTracker::~TaskTracker(){
    // delete the pools
    delete Mapper_threads;
    delete Reducer_threads;
}