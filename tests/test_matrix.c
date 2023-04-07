#include "matrix.h"
#include <stdio.h>

void print_matrix(Matrix *m) {
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            printf("%.2f ", m->data[i*m->cols+j]);
        }
        printf("\n");
    }
}

int main() {
    // Test matrix creation
    float dataA[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    Matrix *A = create_matrix_from_array(3, 3, (float *)dataA);
    
    printf("Matrix A:\n");
    print_matrix(A);

    // Test matrix addition
    float dataB[3][3] = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };
    Matrix *B = create_matrix_from_array(3, 3, (float *)dataB);
    printf("Matrix B:\n");
    print_matrix(B);

    Matrix *C = create_matrix(3, 3);
    matrix_add(C, A, B);
    printf("Matrix A + B:\n");
    print_matrix(C);

    // Test matrix subtraction
    matrix_subtract(C, A, B);
    printf("Matrix A - B:\n");
    print_matrix(C);

    // Test matrix multiplication
    matrix_multiply(C, A, B);
    printf("Matrix A * B:\n");
    print_matrix(C);

    // Test matrix transpose
    matrix_transpose(C, A);
    printf("Matrix A Transpose:\n");
    print_matrix(C);

    // Test matrix padding
    Matrix *D = create_matrix(5, 5);
    matrix_padding(D, A, 1, 1, 1, 1, 0);
    printf("Matrix A with padding:\n");
    print_matrix(D);

    // Test matrix convolution
    // Define kernel matrix
    float kernel_data[2][2] = {
        {1, 0}, 
        {0, 1}
    };
    Matrix *kernel = create_matrix_from_array(2, 2, (float *)kernel_data);

    // Define output matrix
    Matrix *output = create_matrix(4, 4);

    // Perform convolution
    matrix_convolve(output, A, kernel, 1, 1);

    // Print the output matrix
    printf("Convolution result:\n");
    print_matrix(output);

    // Test Matrix flip
    Matrix *output_flip = create_matrix(4, 4);
    matrix_flip(output_flip, output);
    printf("Flipped result:\n");
    print_matrix(output_flip);

    // Free memory
    free_matrix(output_flip);
    free_matrix(kernel);
    free_matrix(output);
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(D);

    return 0;
}
