#include "ndarray.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void test_cal(){
    ndarray *a = nda_zero(3, (int[]){2, 3, 2});
    ndarray *b = nda_zero(3, (int[]){2, 3, 2});
    ndarray *out = nda_zero(3, (int[]){2, 3, 2});

    nda_init_data(a, (float[]){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    nda_init_data(b, (float[]){5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15, 16});

    nda_add(a, b, out);
    // nda_print_mat(out);
    printf("\n");
    
    nda_free(a);
    nda_free(b);
    nda_free(out);
}

void test_conv2d(){
    ndarray *a = nda_zero(2, (int[]){3, 3});
    ndarray *b = nda_zero(2, (int[]){2, 2});
    ndarray *out = nda_zero(2, (int[]){2, 2});

    nda_init_data(a, (float[]){1, 2, 3, 4, 5, 6, 7, 8, 9});
    nda_init_data(b, (float[]){1, 0, 0, 1});

    nda_conv2d(a, b, out);
    nda_print_mat(a);
    nda_print_mat(b);
    nda_print_mat(out);
    printf("\n");
    nda_free(a);
    nda_free(b);
    nda_free(out);
}

void test_reshape(){
    ndarray *a = nda_zero(3, (int[]){2, 3, 2});
    nda_init_rand(a);
    // nda_print_mat(a);
    printf("\n");

    nda_reshape(a, 2, (int[]){3, 4});
    nda_print_mat(a);
    printf("\n");

    nda_free(a);
}

// void test_conv3d(){
//     ndarray *a = nda_zero(3, (int[]){3, 3, 3});
//     ndarray *b = nda_zero(4, (int[]){1, 3, 2, 2});
//     ndarray *out = nda_zero(3, (int[]){1, 2, 2});

//     nda_init_data(a, (float[]){1 , 2, 3, 4, 5, 6, 7, 8, 9, 
//                                10, 11, 12, 13, 14, 15, 16, 17, 18, 
//                                19, 20, 21, 22, 23, 24, 25, 26, 27});
//     nda_init_data(b, (float[]){1, 0, 0, 1,
//                                1, 0, 0, 1,
//                                1, 0, 0, 1});

//     nda_conv3d(a, b, out);
//     nda_print_mat(a);
//     nda_print_mat(b);
//     nda_print_mat(out);
//     printf("\n");

//     nda_free(a);
//     nda_free(b);
//     nda_free(out);
// }

void test_conv3d_time(){
    ndarray *a = nda_zero(3, (int[]){32, 6, 5});
    ndarray *b = nda_zero(4, (int[]){64, 32, 2, 2});
    ndarray *out = nda_zero(3, (int[]){64, 5, 4});

    clock_t start, end;
    int total = 0;
    int times = 100;
    for (int i = 0; i < times; i++){
        nda_init_rand(a);
        nda_init_rand(b);
        start = clock();
        nda_conv3d(a, b, out);
        end = clock();
        total += (end - start);
    }
    
    printf("Time: %f\n", (double)(total) / CLOCKS_PER_SEC / times);
    nda_free(a);
    nda_free(b);
    nda_free(out);
}

void test_transpose(){
    ndarray *a = nda_zero(2, (int[]){2, 3});

    nda_init_data(a, (float[]){1, 2, 3, 4, 5, 6});
    nda_print_mat(a);
    printf("\n");
    nda_T(a);
    nda_print_mat(a);
    printf("\n");

    nda_free(a);
}

void test_flip(){
    ndarray *a = nda_zero(2, (int[]){3, 3});

    nda_init_data(a, (float[]){1, 2, 3, 4, 5, 6, 7, 8, 9}); nda_print_mat(a); printf("\n"); 
    nda_flip(a); nda_print_mat(a); printf("\n"); 
    nda_flip(a); nda_print_mat(a); printf("\n"); 
    nda_T(a); nda_print_mat(a); printf("\n"); 
    nda_flip(a); nda_print_mat(a); printf("\n"); 
    nda_T(a); nda_print_mat(a); printf("\n");

    nda_free(a);
}

int main() {
    // srand(time(NULL));
    // test_cal();
    // test_conv2d();
    // test_conv3d();
    // test_conv3d_time();
    // test_reshape();
    // test_transpose();
    test_flip();
    return 0;
}