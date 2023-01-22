#pragma once

#include <pthread.h>
#include <queue>

namespace MapReduce
{

class ThreadPool;
class ThreadPoolTask;

class ThreadPoolTask {
    friend class ThreadPool;
public:
    ThreadPoolTask(void* (*f)(void*), void* arg):
        f(f),
        arg(arg)
    {}

private:
    void* (*f)(void*);
    void* arg;
};

class ThreadPool {
public:
    ThreadPool(int numThreads) :
        _numThreads(numThreads)
    {
        _threads = new pthread_t[_numThreads];
        _terminate = false;

        pthread_mutex_init(&_mutex, nullptr);
        pthread_cond_init(&_addCond, nullptr);
        pthread_cond_init(&_removeCond, nullptr);
    }

    ~ThreadPool() {
        pthread_cond_destroy(&_addCond);
        pthread_cond_destroy(&_removeCond);
        pthread_mutex_destroy(&_mutex);

        while (!_tasks.empty()) {
            auto task = _tasks.front();
            _tasks.pop();


            delete task;
        }

        delete[] _threads;
    }

    bool getTerminate() {
        return _terminate;
    }

    static void* run(void *arg);
    ThreadPoolTask* removeTask();
    void addTask(ThreadPoolTask* task);
    void start();
    void join();
    void terminate();

private:
    int _numThreads;
    bool _terminate;
    const size_t _bufferSize = 1;

    pthread_t* _threads;
    pthread_mutex_t _mutex;
    pthread_cond_t _removeCond;
    pthread_cond_t _addCond;
    std::queue<ThreadPoolTask*> _tasks;
};

void* ThreadPool::run(void *arg) {
    ThreadPool *pool = (ThreadPool*)arg;

    while (!pool->_terminate) {
        ThreadPoolTask* task = pool->removeTask();

        if (task != nullptr) {
            (*(task->f))(task->arg);

            delete task;
        }
    }

    return nullptr;
}

ThreadPoolTask* ThreadPool::removeTask() {
    pthread_mutex_lock(&_mutex);

    while (_tasks.empty() and !_terminate) {
        pthread_cond_wait(&_removeCond, &_mutex);
    }

    if (_terminate) {
        pthread_mutex_unlock(&_mutex);
        return nullptr;
    }

    ThreadPoolTask* nextTask = _tasks.front();
    _tasks.pop();
    pthread_cond_signal(&_addCond);

    pthread_mutex_unlock(&_mutex);

    return nextTask;
}

void ThreadPool::addTask(ThreadPoolTask* task) {
    pthread_mutex_lock(&_mutex);

    while (_tasks.size() >= _bufferSize and !_terminate) {
        pthread_cond_wait(&_addCond, &_mutex);
    }

    if (_terminate) {
        pthread_mutex_unlock(&_mutex);
        return;
    }

    _tasks.push(task);
    pthread_cond_signal(&_removeCond);

    pthread_mutex_unlock(&_mutex);
}

void ThreadPool::start() {
    for (int i = 0; i < _numThreads; ++i) {
        pthread_create(&_threads[i], 0, ThreadPool::run, (void*)this);
    }
}

void ThreadPool::join() {
    for (int i = 0; i < _numThreads; ++i) {
        pthread_join(_threads[i], nullptr);
    }
}

void ThreadPool::terminate() {
    pthread_mutex_lock(&_mutex);

    _terminate = true;
    pthread_cond_broadcast(&_addCond);
    pthread_cond_broadcast(&_removeCond);

    pthread_mutex_unlock(&_mutex);
}

};
