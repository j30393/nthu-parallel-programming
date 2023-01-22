CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 -march=native
hw2a: CFLAGS += -pthread
# hw2a: CFLAGS += -fsanitize=address -g 
hw2a: CFLAGS += -fno-math-errno -fno-rounding-math -fno-signaling-nans -fcx-limited-range -fno-signed-zeros -fno-trapping-math -frename-registers
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -fexpensive-optimizations -ffp-contract=fast -fno-math-errno -fno-rounding-math -fno-signaling-nans -fcx-limited-range -fno-signed-zeros -fno-trapping-math -frename-registers
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
