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

int* image;
bool flag; // to show which is more 
// if the row number is more than the column , the flag will be 1
// otherwise flag = 0
int iters , width , height , limit , cnt = -1; // limit -> the threshold of overall 
double left , right , lower , upper; // width = column num
double height_gap , width_gap ;
pthread_mutex_t count_mutex;

void* calculate(void* threadid){
    int* tid = (int*)threadid;
    // row number > column num
    // printf("Create the threat %d \n", *tid);
    // printf("The limit is %d \n",limit);
    int local_cnt;
    while(cnt < limit - 1){
        pthread_mutex_lock(&count_mutex);
        local_cnt = ++cnt;
        /*printf("The limit is %d and the ",limit);
        printf("thread #%d! ", *tid);
        printf("get the %d work \n",cnt);*/
        pthread_mutex_unlock(&count_mutex);
        if(flag){
            // printf("The height > width \n");
            // printf("The width is %d and the height is %d \n",  width ,height);
            while(1){
                double y0 = local_cnt * height_gap + lower;
                for (int i = 0; i < width; ++i){
                    double x0 = i * width_gap + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    // printf("repeats : %d \n" , repeats);
                    image[local_cnt * width + i] = repeats;
                }
                break;
            }
            // width -> cnt 
        }
        else{
            // height -> cnt
            // printf("The height <= width \n");
            // printf("The width is %d and the height is %d \n",  width ,height);
            while(1){
                double x0 = local_cnt * width_gap + left;
                for (int j = 0; j < height; ++j){
                    double y0 = j * height_gap + lower;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                        // printf("repeats : %d \n" , repeats);
                    }
                    image[j * width + local_cnt] = repeats;
                }
                break;
            }
            // pthread_mutex_lock(&mutex);
            // pthread_mutex_unlock(&mutex);
        }
    }
    pthread_exit(NULL);

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

    // set the flag's value
    flag = (height > width);
    if(flag) limit = height;
    else limit = width;
    // printf("hello \n");
    height_gap = (upper - lower) / height;
    width_gap = (right - left) / width;
    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);
    pthread_t threads[cpu_num];
    int ID[cpu_num];

    for(int i = 0 ; i < cpu_num ; i++){
        ID[i] = i;
        // printf("In main: creating thread %d\n", i);
        pthread_create(&threads[i], NULL, calculate, (void*)&ID[i]);
    }

    for(int i = 0; i < cpu_num ; i++){
		pthread_join(threads[i], NULL);
	}

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
    pthread_mutex_destroy(&count_mutex);
}
