#include "matrix.h"
#include <stdlib.h>
#include <string.h>

/* Constructor and destructor */
Matrix *create_matrix(size_t rows, size_t cols) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = (float *) calloc(rows * cols, sizeof(float));
    return matrix;
}

Matrix *create_matrix_from_array(size_t rows, size_t cols, float *data){
    Matrix *matrix = create_matrix(rows, cols);
    memcpy(matrix->data, data, rows * cols * sizeof(float));
    return matrix;
}

void free_matrix(Matrix *matrix) {
    free(matrix->data);
    free(matrix);
}

/* Helper functions for size checking */
void assert_same_size(Matrix *a, Matrix *b) {
    assert(a->rows == b->rows && a->cols == b->cols && "Matrix dimensions must match.");
}

void assert_multipliable(Matrix *result, Matrix *a, Matrix *b) {
    assert(a->cols == b->rows && "Matrix dimensions must be compatible for multiplication.");
    assert(result->rows == a->rows && result->cols == b->cols && "Result matrix dimensions must match the dimensions of the product.");
}

/* Matrix operations */
void matrix_reset(Matrix *matrix) {
    memset(matrix->data, 0, matrix->rows * matrix->cols * sizeof(float));
}

void matrix_multiply(Matrix *result, Matrix *a, Matrix *b) {
    assert_multipliable(result, a, b);
    matrix_reset(result);

    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < b->cols; j++) {
            for (size_t k = 0; k < a->cols; k++) {
                result->data[i * result->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }
}

void matrix_transpose(Matrix *result, Matrix *a) {
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            result->data[j * result->cols + i] = a->data[i * a->cols + j];
        }
    }
}

void matrix_add(Matrix *result, Matrix *a, Matrix *b) {
    assert_same_size(a, b);
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            result->data[i * result->cols + j] = a->data[i * a->cols + j] + b->data[i * b->cols + j];
        }
    }
}

void matrix_subtract(Matrix *result, Matrix *a, Matrix *b) {
    assert_same_size(a, b);
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            result->data[i * result->cols + j] = a->data[i * a->cols + j] - b->data[i * b->cols + j];
        }
    }
}

void matrix_elementwise_multiply(Matrix *result, Matrix *a, Matrix *b) {
    assert_same_size(a, b);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
            result->data[i] = a->data[i] * b->data[i];
    }
}

void matrix_elementwise_divide_with_epsilon(Matrix *result, Matrix *a, Matrix *b, float epsilon) {
    assert_same_size(a, b);
    for (size_t i = 0; i < a->rows * a->cols; i++) {
            result->data[i] = a->data[i] / (b->data[i] + epsilon);
    }
}

void matrix_scalar_multiply(Matrix *result, Matrix *a, float scalar) {
    for (size_t i = 0; i < a->rows * a->cols; i++) {
            result->data[i] = a->data[i] * scalar;
    }
}

void matrix_convolve(Matrix *result, Matrix *input, Matrix *kernel, int padding, int stride) {
    assert(input && kernel && result);
    assert(stride > 0);

    int padded_rows = input->rows + 2 * padding;
    int padded_cols = input->cols + 2 * padding;

    Matrix *padded_input = create_matrix(padded_rows, padded_cols); 
    matrix_padding(padded_input, input, padding, padding, padding, padding, 0);

    size_t result_rows = (padded_rows - kernel->rows) / stride + 1; //  N=(W-F+2P)/S+1
    size_t result_cols = (padded_cols - kernel->cols) / stride + 1;

    assert(result->rows == result_rows && result->cols == result_cols);

    for (size_t i = 0; i < result_rows; i++) {
        for (size_t j = 0; j < result_cols; j++) {
            float sum = 0.0f;
            for (size_t m = 0; m < kernel->rows; m++) {
                for (size_t n = 0; n < kernel->cols; n++) {
                    sum += padded_input->data[(i * stride + m) * padded_input->cols + (j * stride + n)] * kernel->data[m * kernel->cols + n];
                }
            }
            result->data[i * result->cols + j] = sum;
        }
    }

    free_matrix(padded_input);
}

void matrix_normalize(Matrix *matrix) {
    float min_val = matrix->data[0];
    float max_val = matrix->data[0];

    // Find the minimum and maximum values in the matrix
    for (size_t i = 0; i < matrix->rows * matrix->cols; i++) {
        float value = matrix->data[i];
        if (value < min_val) {
            min_val = value;
        }
        if (value > max_val) {
            max_val = value;
        }
    }

    // Normalize the matrix by subtracting the minimum value and dividing by the range
    float range = max_val - min_val;
    for (size_t i = 0; i < matrix->rows * matrix->cols; i++) {
        matrix->data[i] = (matrix->data[i] - min_val) / range;
    }
}

void matrix_sum_axis(Matrix *result, Matrix *input, size_t axis){
    assert(axis == 0 || axis == 1);
    if (axis == 0) {
        assert(result->rows == 1 && result->cols == input->cols);
        for (size_t i = 0; i < input->cols; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < input->rows; j++) {
                sum += input->data[j * input->cols + i];
            }
            result->data[i] = sum;
        }
    } else {
        assert(result->rows == input->rows && result->cols == 1);
        for (size_t i = 0; i < input->rows; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < input->cols; j++) {
                sum += input->data[i * input->cols + j];
            }
            result->data[i * result->cols] = sum;
        }
    }
}

/* Matrix fitting */
void matrix_padding(Matrix *result, Matrix *input, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right, float pad_value) {
    assert(result->rows == input->rows + pad_top + pad_bottom);
    assert(result->cols == input->cols + pad_left + pad_right);
    
    for (size_t i = 0; i < result->rows; i++) {
        for (size_t j = 0; j < result->cols; j++) {
            if (i < pad_top || i >= input->rows + pad_top || j < pad_left || j >= input->cols + pad_left) {
                result->data[i * result->cols + j] = pad_value;
            } else {
                result->data[i * result->cols + j] = input->data[(i - pad_top) * input->cols + (j - pad_left)];
            }
        }
    }
}

void matrix_reshape(Matrix *matrix, size_t rows, size_t cols) {
    assert(matrix->rows * matrix->cols == rows * cols);
    matrix->rows = rows;
    matrix->cols = cols;
}