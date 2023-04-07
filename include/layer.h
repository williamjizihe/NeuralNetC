#ifndef LAYER_H
#define LAYER_H

#include <stddef.h>
#include "matrix.h"
#include "activation.h"

typedef enum {
    DENSE,
    CONVOLUTION,
    MAX_POOLING
    // Add other layer types as needed
} LayerType;

typedef struct Layer {
    LayerType type;
    size_t epoch;
    MatrixSize input_size;
    MatrixSize output_size;
    Matrix *weights;
    Matrix *biases;
    Matrix *output;
    Matrix *weight_gradient;
    Matrix *bias_gradient;
    Matrix *accumulated_gradient;
    Matrix *accumulated_moment;
    ActivationFunction *activation_function;
    OptimizerSpec *optimizer;
    size_t *size_t_params;
    void (*forward)(LayerSpec *current, const Matrix *input);
    void (*backward)(LayerSpec *current, const Matrix *output_gradient, Matrix *input_gradient);
    LayerSpec *next;
    LayerSpec *prev;
} LayerSpec;

// Function prototypes
LayerSpec *create_dense_layer(LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_function_type);

LayerSpec *create_convolution_layer(LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_function_type, size_t kernel_rows, size_t kernel_cols, size_t padding, size_t stride);

LayerSpec *create_max_pooling_layer(LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_function_type, size_t kernel_rows, size_t kernel_cols, size_t padding, size_t stride);

void free_layer(LayerSpec *layer);

#endif // LAYER_H
