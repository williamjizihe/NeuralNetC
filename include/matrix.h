#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <assert.h>

typedef struct MatrixSize {
    size_t rows;
    size_t cols;
} MatrixSize;

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

// Constructor and destructor
Matrix *create_matrix(size_t rows, size_t cols);
Matrix *create_matrix_from_array(size_t rows, size_t cols, float *data);
void free_matrix(Matrix *matrix);

// Helper functions for size checking
void assert_same_size(Matrix *a, Matrix *b);
void assert_multipliable(Matrix *result, Matrix *a, Matrix *b);

// Matrix operations
void matrix_reset(Matrix *matrix);
void matrix_multiply(Matrix *result, Matrix *a, Matrix *b);
void matrix_transpose(Matrix *result, Matrix *a);
void matrix_add(Matrix *result, Matrix *a, Matrix *b);
void matrix_subtract(Matrix *result, Matrix *a, Matrix *b);
void matrix_elementwise_multiply(Matrix *result, Matrix *a, Matrix *b);
void matrix_elementwise_divide_with_epsilon(Matrix *result, Matrix *a, Matrix *b, float epsilon);
void matrix_scalar_multiply(Matrix *result, Matrix *a, float scalar);
void matrix_convolve(Matrix *result, Matrix *input, Matrix *kernel, int padding, int stride); 
void matrix_normalize(Matrix *matrix);
void matrix_sum_axis(Matrix *result, Matrix *input, size_t axis);

// Matrix fitting
void matrix_padding(Matrix *result, Matrix *input, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right, float pad_value);
void matrix_reshape(Matrix *matrix, size_t rows, size_t cols);
void matrix_flip(Matrix *result, Matrix *input);

#endif // MATRIX_H
