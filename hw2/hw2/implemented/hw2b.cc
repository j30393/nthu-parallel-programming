#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <mpi.h>

// for the SIMD instruction
#include <emmintrin.h>
#define _mul_pd(a,b) _mm_mul_pd(a,b)
#define _add_pd(a,b) _mm_add_pd(a,b)
#define _load1_pd(a) _mm_load1_pd(a)
#define _greater(a,b) _mm_cmpgt_pd(a,b)
int* image;
bool flag; // to show which is more 
// if the row number is more than the column , the flag will be 1
// otherwise flag = 0
int iters , width , height , limit , cnt = -1; // limit -> the threshold of overall 
double left , right , lower , upper; // width = column num
double height_gap , width_gap ;
pthread_mutex_t count_mutex;
bool remainder_;
int size;
// for the mpi 
int numtasks, rank, rc;

int calculate_the_rest(int cur_iter , double x0 , double y0 , double x , double y){
    double length_squared = 0;
    /* this cause 440 when using sse2 
    while (cur_iter < iters && length_squared < 4) {
        double temp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = temp;
        length_squared = x * x + y * y;
        ++cur_iter;
    }
    return cur_iter;*/
    double x_sq = x * x ;
    double y_sq = y * y ;
    while (cur_iter < iters && length_squared < 4) {
        y = 2 * x * y + y0;
        x =  x_sq - y_sq + x0;
        x_sq = x * x;
        y_sq = y * y;
        length_squared = x_sq + y_sq;
        ++cur_iter;
    }
    return cur_iter;
}


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int cpu_num = CPU_COUNT(&cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    pthread_mutex_init(&count_mutex, NULL);
    /* argument parsing */

    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10); // column num
    height = strtol(argv[8], 0, 10); // row num

    // printf("hello \n");
    height_gap = (upper - lower) / height;
    width_gap = (right - left) / width;
    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    for(int i = 0; i < width * height; i++){
        image[i] = 0;   // for MPI_Reduce
    }
    assert(image);
    // starting the mpi
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int numtasks_twice = numtasks * 2;
    int rank_twice = rank * 2;

    // set the flag's value
    flag = (height > width);
    int jump_phase = 2*numtasks;
    double zero = 0.00000000000000;
    double one = 1.000000000000000;
    double two = 2.000000000000000;
    double four = 4.0000000000000000;
    double iter_limit = iters;
    __m128d sse_height_gap = _load1_pd(&height_gap);
    __m128d sse_width_gap = _load1_pd(&width_gap);
    __m128d digit_2 = _load1_pd(&two); // (2,2)
    __m128d digit_1 = _load1_pd(&one); // (1,1)
    __m128d digit_0 = _load1_pd(&zero); // (0,0)
    __m128d digit_4 = _load1_pd(&four); // (4,4)
    __m128d iter = _load1_pd(&iter_limit);
    if(flag){
        limit = height;
        remainder_ = ( width%2 == 1 );
        size = (width >> 1 ) << 1;
        #pragma omp parallel for schedule(dynamic , 1)
        for(int i = rank ; i < limit ; i+= numtasks){
            int local_cnt = i;
            double y0 = local_cnt * height_gap + lower;
            __m128d sse_y0 = _load1_pd(&y0); // (y0,y0)
            for(int j = 0 ; j < size ; j+=2){
                // printf("The iter of height %d and y0 : %lf with rank %d \n " , i , y0 , rank);
                __m128d x_init , y_init , length_squared , repeats;
                x_init = y_init = length_squared = repeats = digit_0; // initialize repeat , x, y , length_squared
                __m128d sse_x0 ;
                sse_x0[0] = j*width_gap + left;
                sse_x0[1] = (j + 1)*width_gap + left;
                // both < iter and length_squared < 4
                __m128d x_square = _mul_pd(x_init , x_init);
                __m128d y_square = _mul_pd(y_init , y_init);
                while(1){
                    __m128d cmp_ans = _greater(iter , repeats);
                    __m128d cmp_ans_2 = _greater(digit_4 , length_squared);
                    if(cmp_ans[0] != 0 && cmp_ans[1] != 0 && cmp_ans_2[0] != 0 && cmp_ans_2[1] != 0){
                        y_init = _mm_add_pd(sse_y0,_mul_pd(digit_2 ,_mul_pd(x_init , y_init)));
                        x_init = _mm_add_pd(sse_x0,_mm_sub_pd(x_square,y_square));
                        x_square = _mul_pd(x_init , x_init);
                        y_square = _mul_pd(y_init , y_init);
                        length_squared = _mm_add_pd(x_square,y_square);
                        repeats = _mm_add_pd(repeats,digit_1);
                    }
                    else if(cmp_ans[0] != 0 && cmp_ans_2[0] != 0){ // second done
                        int cnt_iter = (int)repeats[0];
                        int final_iter = calculate_the_rest(cnt_iter , sse_x0[0] , sse_y0[0] , x_init[0] , y_init[0]);
                        image[local_cnt * width + j] = final_iter;
                        image[local_cnt * width + j + 1] = (int)repeats[1];
                        break;
                    }
                    else if(cmp_ans[1] != 0 && cmp_ans_2[1] != 0){
                        int cnt_iter = (int)repeats[1];
                        int final_iter = calculate_the_rest(cnt_iter , sse_x0[1] , sse_y0[1] , x_init[1] , y_init[1]);
                        image[local_cnt * width + j] = (int)repeats[0];
                        image[local_cnt * width + j + 1] = final_iter;
                        break;
                    }
                    else{
                        image[local_cnt * width + j] = (int)repeats[0];
                        image[local_cnt * width + j + 1] = (int)repeats[1];
                        break;
                    }
                }
            }
            if(remainder_){
                double y0 = local_cnt * height_gap + lower;
                double x0 = size * width_gap + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                double x_sq = 0 ;
                double y_sq = 0 ;
                while (repeats < iters && length_squared < 4) {
                    y = 2 * x * y + y0;
                    x =  x_sq - y_sq + x0;
                    x_sq = x * x;
                    y_sq = y * y;
                    length_squared = x_sq + y_sq;
                    ++repeats;
                }
                // printf("repeats : %d \n" , repeats);
                image[local_cnt * width + size ] = repeats;
            }
        }
    }
    else {
        limit = width;
        remainder_ = ( height%2 == 1 );
        size = (height >> 1 ) << 1;
        #pragma omp parallel for schedule(dynamic , 1)
        for(int i = rank ; i < limit ; i+= numtasks){
            int local_cnt = i;
            double x0 = local_cnt * width_gap + left;
            __m128d sse_x0 = _load1_pd(&x0); 
            for(int j = 0 ; j < size ; j +=2){
                __m128d x_init , y_init , length_squared , repeats;
                x_init = y_init = length_squared = repeats = digit_0; // initialize repeat , x, y , length_squared
                __m128d sse_y0 ;
                sse_y0[0] = j * height_gap + lower;
                sse_y0[1] = (j + 1) * height_gap + lower;
                // both < iter and length_squared < 4
                __m128d x_square = _mul_pd(x_init , x_init);
                __m128d y_square = _mul_pd(y_init , y_init);
                while(1){
                    __m128d cmp_ans = _greater(iter , repeats);
                    __m128d cmp_ans_2 = _greater(digit_4 , length_squared);
                    if((cmp_ans[0] != 0 && cmp_ans[1] != 0) && (cmp_ans_2[0] != 0 && cmp_ans_2[1] != 0)){
                        y_init = _mm_add_pd(sse_y0,_mul_pd(digit_2 ,_mul_pd(x_init , y_init)));
                        x_init = _mm_add_pd(sse_x0,_mm_sub_pd(x_square,y_square));
                        x_square = _mul_pd(x_init , x_init);
                        y_square = _mul_pd(y_init , y_init);
                        length_squared = _mm_add_pd(x_square,y_square);
                        repeats = _mm_add_pd(repeats,digit_1);
                    }
                    else if(cmp_ans[0] != 0 && cmp_ans_2[0] != 0){ // second done
                        int cnt_iter = (int)repeats[0];
                        int final_iter = calculate_the_rest(cnt_iter , sse_x0[0] , sse_y0[0] , x_init[0] , y_init[0]);
                        image[ j * width + local_cnt] = final_iter;
                        image[ (j + 1 ) * width + local_cnt ] = (int)repeats[1];
                        break;
                    }
                    else if(cmp_ans[1] != 0 && cmp_ans_2[1] != 0){
                        int cnt_iter = (int)repeats[1];
                        int final_iter = calculate_the_rest(cnt_iter , sse_x0[1] , sse_y0[1] , x_init[1] , y_init[1]);
                        image[ j * width + local_cnt] = (int)repeats[0];
                        image[ (j + 1 ) * width +local_cnt ] = final_iter;
                        break;
                    }
                    else{
                        image[j * width + local_cnt] = (int)repeats[0];
                        image[(j + 1 ) * width + local_cnt ] = (int)repeats[1];
                        break;
                    }
                }
            }
            if(remainder_){
                double y0 = size * height_gap + lower;
                double x0 = local_cnt * width_gap + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                double x_sq = 0 ;
                double y_sq = 0 ;
                while (repeats < iters && length_squared < 4) {
                    y = 2 * x * y + y0;
                    x =  x_sq - y_sq + x0;
                    x_sq = x * x;
                    y_sq = y * y;
                    length_squared = x_sq + y_sq;
                    ++repeats;
                }
                // printf("repeats : %d \n" , repeats);
                image[size * width + local_cnt ] = repeats;
            }
        }
    }

    /*for(int i = 0 ; i < height ; i++){
        for(int j = 0 ; j < width ; j++){
            printf("%d " , image[i*width + j]);
        }
        printf("\n");
    }*/

    /* draw and cleanup */
    int* final_image = (int*)malloc(width * height * sizeof(int));
    MPI_Reduce(image, final_image, width*height, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0){
        write_png(filename, iters, width, height, final_image);
    }
    free(image);
    free(final_image);
    pthread_mutex_destroy(&count_mutex);
    MPI_Finalize();
}
