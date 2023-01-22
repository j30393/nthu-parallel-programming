CC = mpicc
CXX = mpicxx
CFLAGS = -O3 -lm -pthread
CXXFLAGS = -O3 -std=c++11 -pthread

SRC_DIR = ./
INCLUDE = ./
OUT_DIR = ./

SOURCES := $(wildcard $(SRC_DIR)/*.cc)
OBJECTS := $(patsubst $(SRC_DIR)/%.cc, $(OUT_DIR)/%.o, $(SOURCES))
BIN = mapreduce

.PHONY: all clean

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN)

clean:
	rm $(OBJECTS)
	rm $(BIN)
	
%: %.cc
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^


