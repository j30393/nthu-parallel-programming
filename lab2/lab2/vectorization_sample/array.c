#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <pmmintrin.h>

void multiply_and_add(int *a, int *b, int *c, int *d, long long size) {
  // This loop will be vectorized with compiling flag `-O3` automatically
  for (long long i = 0; i < size; i++) {
    a[i] = b[i] * c[i] + d[i];
  }
}

void multiply_and_add_sse42(int *a, int *b, int *c, int *d, long long size) {
  long long size1 = (size >> 2) << 2;
  long long i = 0;
  for (; i < size1; i += 4) {
    // 1. Declare variable to store 128-bit data with 4 integers
    __m128i as;
    __m128i bs = _mm_lddqu_si128((__m128i const*)&b[i]);
    __m128i cs = _mm_lddqu_si128((__m128i const*)&c[i]);
    __m128i ds = _mm_lddqu_si128((__m128i const*)&d[i]);

    // 2. Perform SSE multiply & add instructions
    as = _mm_mullo_epi32(bs, cs);
    as = _mm_add_epi32(as, ds);
    
    // 3. Store result back to memory
    _mm_store_si128((__m128i*)&a[i], as);
  }

  // Dealing with the remaining parts
  for (long long i = size1; i < size; i++) {
    a[i] = b[i] * c[i] + d[i];
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("You should give 4 arguments.\n1. (int) Random seed.\n2. (long long) Size of the 4 testing arrays.\n3. (long long) # iterations you want to perform.\n4. (int) Whether you want to use SSE4.2 instruction set.\n   (0 => no, 1 => yes)\n");
    return 1;
  }

  srand(atoi(argv[1]));
  long long size = atoll(argv[2]);
  long long count = atoll(argv[3]);
  int flag = atoi(argv[4]);
  char *str1 = "multiply_and_add", *str2 = "multiply_and_add_sse42";
  char *str[2] = {str1, str2};

  MPI_Init(&argc, &argv);
  // generate random array
  int *arr1 = (int*)malloc(sizeof(int) * size);
  int *arr2 = (int*)malloc(sizeof(int) * size);
  int *arr3 = (int*)malloc(sizeof(int) * size);
  int *arr4 = (int*)malloc(sizeof(int) * size);

  for (long long i = 0; i < size; i++) {
    arr2[i] = rand() & 0x007F;
    arr3[i] = rand() & 0x007F;
    arr4[i] = rand() & 0x007F;
  }
  
  // calculate 
  double t1 = MPI_Wtime();
  for (long long j = 0; j < count; j++) {
    if (flag == 0)
      multiply_and_add(arr1, arr2, arr3, arr4, size);
    else
      multiply_and_add_sse42(arr1, arr2, arr3, arr4, size);
  }
  printf("Time for calculating %s: %lf\n", str[flag], MPI_Wtime() - t1);
  long long ind = ((long long)rand() * (long long)rand()) % size;
  printf("Sampled result: arr1[%ld] = %d\n", ind, arr1[ind]);

  MPI_Finalize();
  return 0;
}
