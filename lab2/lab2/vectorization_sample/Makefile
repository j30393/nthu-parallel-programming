CC = mpicc
CXX = mpicxx
LDLIBS = -lm
OPTFLAGS = -march=native -fopt-info-vec-all
CXXFLAGS = $(OPTFLAGS)
CFLAGS = $(OPTFLAGS)
TARGETS = array

.PHONY: all
all: $(TARGETS)

O3: array.c
	$(CC) -O3 $(CFLAGS) array.c -o O3

.PHONY: clean
clean:
	rm -f $(TARGETS) O3
