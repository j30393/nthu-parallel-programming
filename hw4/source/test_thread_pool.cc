#include "gtest/gtest.h"
#include <pthread.h>
#include "thread_pool.h"

#define THREAD_NUM 4

pthread_mutex_t mutex;

struct Args {
    Args(int *total, int num) :
        total(total),
        num(num)
    {}

    int* total;
    int num;
};

void* add(void* input) {
    Args* arg = (Args*)input;

    EXPECT_EQ(pthread_mutex_lock(&mutex), 0);
    *arg->total += arg->num;
    EXPECT_EQ(pthread_mutex_unlock(&mutex), 0);

    return nullptr;
}

void* square(void* input) {
    Args* arg = (Args*)input;

    EXPECT_EQ(pthread_mutex_lock(&mutex), 0);
    *arg->total += (arg->num * arg->num);
    EXPECT_EQ(pthread_mutex_unlock(&mutex), 0);

    return nullptr;
}

class ThreadPoolTest: public ::testing::Test
{
protected:
    MapReduce::ThreadPool* pool;

    void SetUp() override {
        EXPECT_EQ(pthread_mutex_init(&mutex, nullptr), 0);

        pool = new MapReduce::ThreadPool(THREAD_NUM);
        pool->start();

        EXPECT_EQ(pool->getTerminate(), false);
    }

    void TearDown() override {
        pool->terminate();
        pool->join();

        EXPECT_EQ(pthread_mutex_destroy(&mutex), 0);
        EXPECT_EQ(pool->getTerminate(), true);
    }
};

TEST_F(ThreadPoolTest, test1) {
    int* total = new int;
    *total = 0;

    for (int i = 0; i < 3; ++i) {
        Args* arg = new Args(total, i);
        pool->addTask(new MapReduce::ThreadPoolTask(&add, (void*)arg));
    }

    sleep(1);
    EXPECT_EQ(*total, 3);
    delete total;
    total = nullptr;
    EXPECT_EQ(total, nullptr);
    EXPECT_FALSE(pool->getTerminate());
}

TEST_F(ThreadPoolTest, test2) {
    int* total = new int;
    *total = 0;

    for (int i = 0; i < 11; ++i) {
        Args* arg = new Args(total, i);
        pool->addTask(new MapReduce::ThreadPoolTask(&add, (void*)arg));
    }

    sleep(1);
    delete total;
    total = nullptr;
    EXPECT_EQ(total, nullptr);
    EXPECT_FALSE(pool->getTerminate());
}

TEST_F(ThreadPoolTest, test3) {
    int* total = new int;
    *total = 0;

    for (int i = 0; i < 4; ++i) {
        Args* arg = new Args(total, i);
        pool->addTask(new MapReduce::ThreadPoolTask(&square, (void*)arg));
    }

    sleep(1);
    EXPECT_EQ(*total, 14);
    delete total;
    total = nullptr;
    EXPECT_EQ(total, nullptr);
    EXPECT_FALSE(pool->getTerminate());
}

TEST_F(ThreadPoolTest, test4) {
    int* total = new int;
    *total = 0;

    for (int i = 0; i < 4; ++i) {
        Args* arg = new Args(total, i);
        pool->addTask(new MapReduce::ThreadPoolTask(&add, (void*)arg));
    }

    sleep(1);
    EXPECT_EQ(*total, 6);
    EXPECT_FALSE(pool->getTerminate());
    pool->terminate();
    EXPECT_TRUE(pool->getTerminate());

    Args* arg = new Args(total, 4);
    pool->addTask(new MapReduce::ThreadPoolTask(&add, (void*)arg));
    sleep(1);
    EXPECT_NE(*total, 10);
    EXPECT_EQ(*total, 6);
    delete total;
    total = nullptr;
    EXPECT_EQ(total, nullptr);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
