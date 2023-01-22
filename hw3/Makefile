CC = gcc
CXX = g++
CXXFLAGS = -O3 -pthread
CFLAGS = -O3 -lm -pthread
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp -use_fast_math
LDFLAGS = -lm
EXES = hw3-1 hw3-2 hw3-3

.PHONY: all
all: $(EXES)

.PHONY: clean
clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
